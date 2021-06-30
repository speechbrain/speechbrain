#!/usr/bin/env python3
'''
 * Basic training parameters for Event Detection DCASE 2019.
 Author:
 * Vishal Ghorpade 
 * Julien Bouvier Tremblay 2021
'''
from __future__ import print_function, absolute_import

import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from utils_dcase2019 import *
import data_label_pipeline
import scipy
import sed_eval
import dcase_util
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_evaluation_metrics(reference_list, prediction_list, class_list, f_measure_only=False):
    """Compute evaluation metric with SED eval toolkit
        Arguments
        ---------
        reference_list : list of dict
            list of ground truth (see example)
        prediction_list : list of dict
            list of event prediction (see example)
        class_list : List
            list of classes names
        f_measure_only : bool
            if true, function will return only f-score macro (event and segment based)

        Returns
        -------
        tuple of metrics scores

        Example of list
        ---------------
        [   
            {  'event_label': 'dog',
                'event_onset': 0.0,
                'event_offset': 2.5,
                'filename': 'b099.wav'},
            {  'event_label': 'cat',
                'event_onset': 1.0,
                'event_offset': 5.5,
                'filename': 'b099.wav'},
        ...]

    """
    reference_event_list = dcase_util.containers.MetaDataContainer(
                reference_list
            )

    estimated_event_list = dcase_util.containers.MetaDataContainer(
                prediction_list
            )

    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(
                event_label_list=class_list,
                time_resolution=1.0
            )
    event_based_metrics = sed_eval.sound_event.EventBasedMetrics(
                event_label_list=class_list,
                t_collar=0.200,
                percentage_of_length=0.2,
                empty_system_output_handling='zero_score'
            )
    for filename in reference_event_list.unique_files:
            reference_event_list_for_current_file = reference_event_list.filter(
                    filename=filename
                )
            estimated_event_list_for_current_file = estimated_event_list.filter(
                    filename=filename
                )
            segment_based_metrics.evaluate(
                    reference_event_list=reference_event_list_for_current_file,
                    estimated_event_list=estimated_event_list_for_current_file
                )
            event_based_metrics.evaluate(
                    reference_event_list=reference_event_list_for_current_file,
                    estimated_event_list=estimated_event_list_for_current_file
                )

    seg_f_measure = segment_based_metrics.results_class_wise_average_metrics()['f_measure']['f_measure']
    event_f_measure = event_based_metrics.results_class_wise_average_metrics()['f_measure']['f_measure']

    if f_measure_only:
        return seg_f_measure, event_f_measure
    else:
        return segment_based_metrics, event_based_metrics

def get_batch_predictions(strong_pred, weak_pred, filenames, decoder, append_list):
    """Compute predictions for each batch during evaluation to be
            use into the MetaDataContainer of SED eval
        Arguments
        ---------
        strong_pred : torch.Tensor or tensors
            strong prediction batch x frame x n_classes
        weak_pred : torch.Tensor or tensors
            weak prediction batch x n_classes
        filenames : List
            list containing names of the files in the batch (batch_size, )
        decoder : function()
            decode strong predictions tensors into list of dict to be use in MetaDataContainer os SED eval
        append_list:
            a list should be given to append the results

        Returns
        -------
        None
    """
    depooling = hparams["pooling_time_ratio"]/(hparams["sample_rate"]/hparams["hop_length"])
    synt_file = hparams["JsonMetaData"]["train"]["synthetic"]
    window_sizes = find_window_sizes(synt_file, depooling, beta=0.333)

    weak_pred = weak_pred.cpu()
    weak_pred = weak_pred.detach().numpy()
    strong_pred = strong_pred.cpu()
    strong_pred = strong_pred.detach().numpy()
   
    weak = dcase_util.data.ProbabilityEncoder().binarization(weak_pred, binarization_type="global_threshold",
                                                        time_axis=0,
                                                        threshold=0.5)
    # print(weak)
    weak = np.expand_dims(weak, 1)
    # print(strong_pred.max(1))
    strong_pred = strong_pred * weak

    for sample, filename in zip(strong_pred, filenames):

        for i, window in enumerate(window_sizes):
            sample[:,i] = scipy.ndimage.filters.median_filter(sample[:,i], (window,))

        sample = dcase_util.data.ProbabilityEncoder().binarization(sample, binarization_type="global_threshold",
                                                        threshold=0.5)

        for i, window in enumerate(window_sizes):
            sample[:,i] = scipy.ndimage.filters.median_filter(sample[:,i], (window,))

        append_list = decoder(sample, filename, depooling, append_list)
    
    return append_list
    

class EventBrain(sb.Brain):
   
    def compute_forward(self, batch, ema=False, stage=None):
        """Forward computations from the mixture to the separated signals
        Arguments
        ---------
        batch : torch.Tensor or tensors
            An element from the dataloader, including inputs for processing.
        ema : Bool
            If true forward pass with the Teacher model instead.
        stage : Stage
            The stage of the experiment: Stage.TRAIN, Stage.VALID, Stage.TEST

        Returns
        -------
        tuple of torch.Tensor or Tensors
            The outputs of the model strong pred and weak pred.
        """
        feat, lens = self.prepare_features(batch["sig"], stage, ema)
        # print(feat.shape)
        
        if ema==False:
            strong, weak = self.modules.crdnn(feat)
        else:
            feat = feat.detach()
            strong, weak = self.modules.crdnn_ema(feat)
            strong = strong.detach()
            weak = weak.detach()

        return strong, weak

    def prepare_features(self, wavs, stage, ema):
        """Prepare the features for computation, including augmentation.
        Arguments
        ---------
        wavs : tuple
            Input signals (tensor) and their relative lengths (tensor).
        stage : sb.Stage
            The current stage of training.
        ema : Bool
            If true compute different env corrupt.

        Returns
        -------
        tuple of torch.Tensor or Tensors
            feature and length
        """
        wavs, lens = wavs
        if (ema==False and stage==sb.Stage.TRAIN):
            wavs_noise = self.modules.env_corrupt_5(wavs, lens)
            feat = self.hparams.compute_features(wavs_noise) #, self.hparams.STFTArgs, self.hparams.FBArgs)
        elif (ema==True and stage==sb.Stage.TRAIN):
            wavs_noise = self.modules.env_corrupt_10(wavs, lens)
            feat = self.hparams.compute_features(wavs_noise)#, self.hparams.STFTArgs, self.hparams.FBArgs)
        else:
            # wavs_noise = self.modules.env_corrupt_15(wavs, lens)
            feat = self.hparams.compute_features(wavs) #, self.hparams.STFTArgs, self.hparams.FBArgs)
            # print("noise added! ")
        feat = self.modules.mean_var_norm(feat, lens)

        return feat, lens

    def compute_objectives(self, stage, target_strong, target_weak, strong_pred, weak_pred, strong_pred_ema=None, weak_pred_ema=None, rampup_value=None, strong_mask=None, weak_mask=None):
        """Computes the loss functions between estimated and ground truth sources
        Arguments
        ---------
        stage : stage : sb.Stage
            The current stage of training.
        target_strong : torch.Tensor or Tensors
            strong ground truth
        target_weak : torch.Tensor or Tensors
            weak ground truth
        strong_pred : torch.Tensor or Tensors
            strong prediction
        weak_pred : torch.Tensor or Tensors
            weak prediction
        strong_pred_ema : torch.Tensor or Tensors
            strong prediction of Teacher (default = None)
        weak_pred_ema : torch.Tensor or Tensors
            weak prediction of Teacher (default = None)
        rampup_value : float
            ramp-up value (default = None)
        strong_mask : numpy.array
            mask to get only strongly annotated data (default = None)
        weak_mask : numpy.array
            mask to get only weakly annotated data (default = None)
        Returns
        -------
        torch.Tensor or Tensors
            loss
        """
        loss = 0

        self.hparams.loss = self.hparams.loss.cuda()
        self.hparams.consistency_loss = self.hparams.consistency_loss.cuda()
        
        if weak_mask is not None:
            weak_class_loss = self.hparams.loss(weak_pred[weak_mask], target_weak[weak_mask])
            if weak_pred_ema is not None:
                weak_ema_class_loss = self.hparams.loss(weak_pred_ema[weak_mask], target_weak[weak_mask])
                self.hparams.weak_ema_loss = self.update_average(weak_ema_class_loss, self.hparams.weak_ema_loss)

            self.hparams.weak_loss = self.update_average(weak_class_loss, self.hparams.weak_loss)
            
            loss += weak_class_loss
            
        if strong_mask is not None:
            strong_class_loss = self.hparams.loss(strong_pred[strong_mask], target_strong[strong_mask])
            if strong_pred_ema is not None:
                strong_ema_class_loss = self.hparams.loss(strong_pred_ema[strong_mask], target_strong[strong_mask])
                self.hparams.strong_ema_loss = self.update_average(weak_ema_class_loss, self.hparams.strong_ema_loss)
            
            self.hparams.strong_loss = self.update_average(strong_class_loss, self.hparams.strong_loss)
            
            loss += strong_class_loss

        if (self.hparams.Unlabel==1 and stage==sb.Stage.TRAIN):

            consistency_cost = self.hparams.max_consistency_cost * rampup_value
            consistency_loss_weak = consistency_cost * self.hparams.consistency_loss(weak_pred, weak_pred_ema)
            self.hparams.consistency_loss_weak = self.update_average(consistency_loss_weak, self.hparams.consistency_loss_weak)
            
            loss += consistency_loss_weak

            consistency_loss_strong = consistency_cost * self.hparams.consistency_loss(strong_pred, strong_pred_ema)
            self.hparams.consistency_loss_strong = self.update_average(consistency_loss_strong, self.hparams.consistency_loss_strong)

            loss += consistency_loss_strong

        return loss

    def fit_batch(self, batch):
        """fit batch
        Arguments
        ---------
        batch : torch.Tensor or tensors
            An element from the dataloader, including inputs for processing.

        Returns
        -------
        detached loss
        """

        batch = batch.to(self.device)
        filenames = batch['filename']
        batch_size = len(filenames)
        if self.hparams.Unlabel ==1:
            mask_weak = np.arange(batch_size//4)
            mask_strong = np.arange(batch_size//4+ batch_size//2, batch_size)
        elif self.hparams.Strong==1:
            mask_weak = np.arange(batch_size//2)
            mask_strong = np.arange(batch_size//2, batch_size)
        else:
            mask_weak = np.arange(batch_size)
            mask_strong = None

        strong_encode = batch['strong_encoded'].data.to(self.device)
        mh = ManyHotEncoder(self.hparams.classes)
        weak_encode = batch['weak_encoded'].data.to(self.device)
        strong, weak = self.compute_forward(batch, ema=False, stage=sb.Stage.TRAIN)
        if self.hparams.Unlabel ==1:
            # forward pass for MT model
            strong_ema, weak_ema = self.compute_forward(batch, ema=True, stage=sb.Stage.TRAIN)
            # computing ramp-up value
            rampup_length = self.hparams.len_train_loader * self.hparams.number_of_epochs // 2
            if (self.step-1) < rampup_length:
                rampup_value = sigmoid_rampup(self.step-1, rampup_length)
            else:
                rampup_value = 1.0
            loss = self.compute_objectives(sb.Stage.TRAIN, strong_encode, weak_encode, strong, weak, strong_ema, weak_ema, rampup_value, mask_strong, mask_weak)
        else:
            loss = self.compute_objectives(sb.Stage.TRAIN, strong_encode, weak_encode, strong, weak, strong_mask=mask_strong, weak_mask=mask_weak)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.hparams.Unlabel==1:
            update_ema_variables(self.modules.crdnn, self.modules.crdnn_ema, 0.999, self.step-1)
        return loss.detach().cpu()
    
    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """

        # Run this *after* starting all processes since jit modules cannot be
        # pickled.
        self._compile_jit()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                device=torch.device(self.device)
            )

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the start of epoch to set all the items to 0 or as empty lists.
        """
        if stage == sb.Stage.TRAIN:
            self.hparams.weak_loss = 0.0
            self.hparams.strong_loss = 0.0
            self.hparams.consistency_loss_weak = 0.0
            self.hparams.consistency_loss_strong = 0.0
            self.hparams.iter_idx = 0

        print(stage)
        print("epoch{0}".format(self.hparams.epoch_counter.current))

        if stage == sb.Stage.VALID:
            self.hparams.ref_event_list = []
            self.hparams.est_event_list = []
            self.hparams.weak_loss = 0.0
            self.hparams.strong_loss = 0.0

        if stage == sb.Stage.TEST:
            self.hparams.ref_event_list = []
            self.hparams.est_event_list = []

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of epoch to save variables
        or print out metrics.
        """
        if stage == sb.Stage.TRAIN:
            self.hparams.train_total_loss = stage_loss
            self.hparams.train_weak_loss = self.hparams.weak_loss
            self.hparams.train_strong_loss = self.hparams.strong_loss
            self.hparams.train_consistency = self.hparams.consistency_loss_strong + self.hparams.consistency_loss_weak

        if stage==sb.Stage.VALID:

            seg_score, event_score = get_evaluation_metrics(self.hparams.ref_event_list,
                                                            self.hparams.est_event_list,
                                                            self.hparams.classes,
                                                            f_measure_only=True)
            stats={
                "total_loss": stage_loss,
                "weak_loss": self.hparams.weak_loss,
                "strong_loss": self.hparams.strong_loss,
                "consistency_loss": self.hparams.consistency_loss_strong + self.hparams.consistency_loss_weak,
                "Segment f_score": seg_score,
                "Event f_score": event_score,
            }

            print("Segment f_score")
            print(seg_score)
            print("Event f_score")
            print(event_score)

            self.hparams.train_logger.log_stats(
                    stats_meta={"epoch": epoch},
                    train_stats={"total_loss": self.hparams.train_total_loss,
                                "weak_loss": self.hparams.train_weak_loss,
                                "strong_loss": self.hparams.train_strong_loss,
                                "consistency_loss": self.hparams.train_consistency
                                },
                    valid_stats={
                        "total_loss": stage_loss,
                        "weak_loss": self.hparams.weak_loss,
                        "strong_loss": self.hparams.strong_loss,
                        "Segment f_score": seg_score,
                        "Event f_score": event_score,
                    },
                )
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=["Event f_score"])

        if stage==sb.Stage.TEST:
            seg_score, event_score = get_evaluation_metrics(self.hparams.ref_event_list,
                                                            self.hparams.est_event_list,
                                                            self.hparams.classes)

            sys.stdout = open(self.hparams.evaluation_metric, "w")
            print("epoch: ", self.hparams.epoch_counter.current)
            print(seg_score)
            print(event_score)
            sys.stdout.close()

    def init_optimizers(self):
        """Called during ``on_fit_start()``, initialize optimizers
        after parameters are fully configured (e.g. DDP, jit).

        This is only to overwrite the optimizer if unlabel are in use
        """

        if self.opt_class is not None:
            # if using MT model we need to overwrite
            if self.hparams.Unlabel==1:
                self.optimizer = self.opt_class(self.modules.crdnn.parameters())
                print("optimizer set only for Student model weights")
            else:
                self.optimizer = self.opt_class(self.modules.parameters())
            

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def evaluate_batch(self, batch, stage):
        """fit batch
        Arguments
        ---------
        batch : torch.Tensor or tensors
            An element from the dataloader, including inputs for processing.
        stage : sb.Stage
            The current stage of training.

        Returns
        -------
        detached loss
        """

        batch = batch.to(self.device)
        filenames = batch['filename']

        # to compute mask if batches are not the same length
        batch_size = len(filenames)

        # we want to evaualte both type on all sample
        mask_weak = np.arange(batch_size)
        mask_strong = np.arange(batch_size)


        strong_encode = batch['strong_encoded'].data.to(self.device)
        strong_truth = batch['strong_truth']
        weak_encode = batch['weak_encoded'].data.to(self.device)
        
        strong, weak = self.compute_forward(batch, False, stage)
        
        if stage != sb.Stage.TRAIN:
            for events in strong_truth:
                for e in events:
                    self.hparams.ref_event_list.append(e)
            mh = ManyHotEncoder(self.hparams.classes)
            self.hparams.est_event_list = get_batch_predictions(strong, weak,filenames, mh.decode_strong, self.hparams.est_event_list)
            loss = self.compute_objectives(stage, strong_encode, weak_encode, strong, weak,strong_mask=mask_strong, weak_mask=mask_weak)
            return loss.detach().cpu()

# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    # set the device
    if hparams['GPUAvailable']==False:
        run_opts['device']='cpu'

    # Download the data if flag is up
    if hparams["download"] ==1:
        download_data.run_download(hparams)

    # create de datasets
    datasets, batch_sampler_2, batch_sampler_3, batch_sampler_2_toy, batch_sampler_3_toy = data_label_pipeline.dataio_prep(hparams)

    # make sure the Teacher Model weights are detach 
    hparams["modules"]["crdnn_ema"].to(run_opts['device'])
    for param in hparams["modules"]["crdnn_ema"].parameters():
        param.detach_()
    hparams["modules"]["crdnn"].to(run_opts['device'])
    for param in hparams["modules"]["crdnn"].parameters():
        param.requires_grad=True

    EventDet = EventBrain(
      modules=hparams["modules"],
      run_opts=run_opts,
      hparams = hparams,
      opt_class= hparams["opt_class"],
      checkpointer=hparams["checkpointer"]
      )
    
    # if unlabel flag is up fit with 3 types of labels
    if (hparams['Unlabel'] ==1):
        print("You are using all types of data")
        EventDet.fit(
        epoch_counter=EventDet.hparams.epoch_counter,
        train_set=datasets["train_weak_synthetic_unlabel"],
        valid_set=datasets["validation"],
        train_loader_kwargs={"batch_sampler":batch_sampler_3, "collate_fn":sb.dataio.batch.PaddedBatch},
        valid_loader_kwargs=hparams["dataloader_valid_options"]
        )

    # if strong flag is up fit with weakly annotated and strongly annonated only
    elif (hparams['Strong'] ==1):
        print("You are using weakly and strongly labelled data")
        EventDet.fit(
        epoch_counter=EventDet.hparams.epoch_counter,
        train_set=datasets["train_weak_synthetic"],
        valid_set=datasets["validation"],
        train_loader_kwargs={"batch_sampler":batch_sampler_2, "collate_fn":sb.dataio.batch.PaddedBatch},
        valid_loader_kwargs=hparams["dataloader_valid_options"]
        )
    
    # if nor strong flag nor unlabel are up fit with only weakly annotated
    else:
        print("You are using only weakly labelled data")
        EventDet.fit(
        epoch_counter=EventDet.hparams.epoch_counter,
        train_set=datasets["train_weak"],
        valid_set=datasets["validation"],
        train_loader_kwargs=hparams["dataloader_train_options"],
        valid_loader_kwargs=hparams["dataloader_valid_options"]
        )

    EventDet.evaluate(
            test_set=datasets["test"],
            test_loader_kwargs=hparams["dataloader_test_options"],
            )
