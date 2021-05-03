# -*- coding: utf-8 -*-
""" runlike this
# python train.py --device=cuda:0 --max_grad_norm=1.0 hparams.yaml	
"""

import torch
from torchaudio import transforms
import speechbrain as sb
import speechbrain
from speechbrain.dataio.dataio import read_audio
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.dataset import DynamicItemDataset
from hyperpyyaml import load_hyperpyyaml
import sys
import os
import logging
from speechbrain.utils.distributed import run_on_main

###
from model import Tacotron2 
from textToSequence import text_to_sequence
##
sys.path.append("..")

logger = logging.getLogger(__name__)

class Tacotron2Brain(sb.Brain):

    def compute_forward(self, batch, stage):
        inputs, y, num_items = batch_to_gpu(batch)
        return self.hparams.model(inputs)  #1#2#

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.
        Arguments
        ---------
        predictions : torch.Tensor
            The model generated spectrograms and other metrics from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        
        inputs, y, num_items = batch_to_gpu(batch)
        return criterion(predictions, y)

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch.
        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """
        
        
        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        # Summarize the statistics from the stage for record-keeping.
        else:
            stats = {
                "loss": stage_loss,
            }
        
        # At the end of validation, we can write
        if stage == sb.Stage.VALID:
            # Update learning rate
            old_lr, new_lr = self.hparams.lr_annealing(stage_loss) #1#2#
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
        
            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats( #1#2#
                {"Epoch": epoch},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )
        
            # Save the current checkpoint and delete previous checkpoints.
            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["loss"])
        
        
        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    data_folder = hparams["data_folder"]

    train_data = DynamicItemDataset.from_json(
        json_path=hparams["json_train"], replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["json_valid"], replacements={"data_root": data_folder},
    )


    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["json_test"], replacements={"data_root": data_folder},
    )

    datasets = [train_data, valid_data, test_data]

    audio_toMel = transforms.MelSpectrogram(
		sample_rate = hparams['sampling_rate'] ,
		
		hop_length = hparams['hop_length'] ,
		win_length = hparams['win_length'] ,
		n_fft=hparams['n_fft'],
		n_mels = hparams['n_mel_channels'] ,
		f_min = hparams['mel_fmin'],
		f_max =hparams['mel_fmax'],
		normalized=hparams['mel_normalized'],
		)

		#  Define audio and text pipeline:
    @speechbrain.utils.data_pipeline.takes("file_path","words")
    @speechbrain.utils.data_pipeline.provides("mel_text_pair")
    def audio_pipeline(file_path,words):
        text_seq = torch.IntTensor(text_to_sequence(words, hparams['text_cleaners']))
        audio = speechbrain.dataio.dataio.read_audio(file_path)
        mel = audio_toMel(audio)
        len_text = len(text_seq)
        yield text_seq,mel,len_text

		# set outputs
    speechbrain.dataio.dataset.add_dynamic_item(datasets, audio_pipeline) 
    speechbrain.dataio.dataset.set_output_keys(
				datasets, ["mel_text_pair"],
			)
    #create dataloaders that are passed to the model.
    train_data_loader = SaveableDataLoader(train_data, batch_size=hparams['batch_size'],collate_fn=TextMelCollate(),drop_last=True )
    valid_data_loader = SaveableDataLoader(valid_data, batch_size=hparams['batch_size'],collate_fn=TextMelCollate(),drop_last=True )
    test_data_loader = SaveableDataLoader(test_data, batch_size=hparams['batch_size'],collate_fn=TextMelCollate(),drop_last=True )

    return train_data_loader, valid_data_loader, test_data_loader
    

#some helper functoions 
def batch_to_gpu(batch):
    text_padded, input_lengths, mel_padded, gate_padded, \
        output_lengths, len_x = batch
    text_padded = to_gpu(text_padded).long()
    input_lengths = to_gpu(input_lengths).long()
    max_len = torch.max(input_lengths.data).item()
    mel_padded = to_gpu(mel_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    output_lengths = to_gpu(output_lengths).long()
    x = (text_padded, input_lengths, mel_padded, max_len, output_lengths)
    y = (mel_padded, gate_padded)
    len_x = torch.sum(output_lengths)
    return (x, y, len_x)

def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x
    
    
 ############loss fucntion
def criterion( model_output, targets):
   mel_target, gate_target = targets[0], targets[1]
   mel_target.requires_grad = False
   gate_target.requires_grad = False
   gate_target = gate_target.view(-1, 1)

   mel_out, mel_out_postnet, gate_out, _ = model_output
   gate_out = gate_out.view(-1, 1)
   mel_loss = torch.nn.MSELoss()(mel_out, mel_target) + \
       torch.nn.MSELoss()(mel_out_postnet, mel_target)
   gate_loss = torch.nn.BCEWithLogitsLoss()(gate_out, gate_target)
   return mel_loss + gate_loss
   
   
### custome Collate function for the dataloader
class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step=1):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        for i in range(len(batch)): #the pipline return a dictionary wiht one elemnent 
            batch[i]=list(batch[i].values())[0]
            
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        # count number of items - characters in text
        len_x = [x[2] for x in batch]
        len_x = torch.Tensor(len_x)
        return text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths, len_x

if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    #########
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    #hparams_file="hparams.yaml"
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    #run_opts={"device ": device}
    
    #############
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
        #hparams = load_hyperpyyaml(fin)

    show_results_every = 5  # plots results every N iterations

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    #sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides
      )

    # Dataset prep
#    from prepare import prepare_FSC  # noqa

    # multi-gpu (ddp) save data preparation
#    run_on_main(
#        prepare_FSC,
#        kwargs={
#            "data_folder": hparams["data_folder"],
#            "save_folder": hparams["output_folder"],
#            "skip_prep": hparams["skip_prep"],
#        },
#    )

    # here we create the datasets objects as well as tokenization and encoding
    train_set, valid_set, test_set = dataio_prepare(hparams)

    # Brain class initialization
    tacotron2_brain = Tacotron2Brain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    tacotron2_brain.fit(
        tacotron2_brain.hparams.epoch_counter,
        train_set,
        valid_set
        
    )

    # Test
    tacotron2_brain.evaluate(test_set)
