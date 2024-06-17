from functools import partial
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from torch.utils.tensorboard import SummaryWriter
import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from eval_utils import *
from speechbrain.dataio.batch import PaddedBatch
import torchaudio
import numpy as np
from data_prep_utils import *
from speechbrain.utils import hpopt as hp

class SEP28kBrain(sb.Brain):
    """Use attentional model to predict words in segments"""
    def compute_feats(self, wavs, lens, stage):
        #All clips should be 16Khz and 3 seconds long thus size 48000
        if(wavs.shape[1]>48000):
                    wavs = wavs[:,:48000]
        elif(wavs.shape[1]<48000):
            pad = torch.zeros([wavs.shape[0], 48000-wavs.shape[1]]).to("cuda:0")
            wavs= torch.cat([wavs,pad], dim=1)
        return wavs

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        waveforms, lens = batch.waveform
        waveforms = self.compute_feats(waveforms, lens, stage)
        bin_out = self.modules.model(waveforms)
        return {"bin_pred" : bin_out}
               

    def compute_objectives(self, predictions, batch, stage):
        labels = batch.label.data
        loss = sb.nnet.losses.bce_loss(predictions["bin_pred"].squeeze(1).float(), labels.squeeze(1).float(), pos_weight=torch.Tensor([self.hparams.positive]).to("cuda:0"))
        binary_preds = torch.round(torch.sigmoid(predictions["bin_pred"])) #torch.argmax(, axis=1)
        self.y_true_binary = torch.cat((self.y_true_binary,labels))
        self.y_preds_binary = torch.cat((self.y_preds_binary,binary_preds))
        return loss


    def on_stage_start(self, stage, epoch):
        "Gets called when a stage (either training, validation, test) starts."
        self.y_preds_binary = torch.tensor(()).to("cuda:0")
        self.y_true_binary = torch.tensor(()).to("cuda:0")
        self.labels = None
    
    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a stage.""" 
        self.compute_metrics(epoch, stage, stage_loss)
        if stage != sb.Stage.TEST:
            writer.add_scalar(f"Loss/{stage.name.split('.')[-1].lower()}", stage_loss, epoch)
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            stage_stats = {"loss": stage_loss}
            stage_stats["macro"] = self.fscore * 100
            
            if stage == sb.Stage.VALID:
                self.results = stage_stats
                self.stage_loss = stage_loss
                
                if self.hparams.ckpt_enable:
                    self.checkpointer.save_and_keep_only(
                        meta=stage_stats, min_keys=["loss"], keep_recent=False, name=f"ckpt_{epoch}"
                    )
                if(stage_loss < self.best_loss):
                    self.best_loss = stage_loss
                    self.best_fscore = self.fscore
                    self.best_epoch = epoch
                stage_stats["best_macro"] = 1-self.best_fscore
                self.hparams.train_logger.log_stats(
                    stats_meta={"epoch": epoch},
                    train_stats={"loss": self.train_loss},
                    valid_stats=stage_stats,
                )
            elif stage == sb.Stage.TEST:
                self.hparams.train_logger.log_stats(
                    stats_meta={"Epoch loaded": self.hparams.counter.current},
                    test_stats=stage_stats,
                )
                self.test_loss = stage_loss
                self.test_macro_fscore = stage_stats["macro"]
                self.test_fscore = self.fscore

    def compute_metrics(self, epoch, stage, stage_loss):
        curr_stage = stage.name.split('.')[-1].lower()
        print(f"******{curr_stage}******")
        self.accuracy, self.fscore, self.missrate, self.cf_matrix, _, _= my_confusion_matrix(self.y_true_binary, self.y_preds_binary)
        print(self.cf_matrix)
        self.hparams.train_logger.log_stats(stats_meta={"\nbin fscore": np.round(self.fscore,4)})
        print("---------------")
        if(self.hparams.num_class>1):
            print(self.cf_multi_matrix)
            self.hparams.train_logger.log_stats(stats_meta={ "\nmulti-fscores": np.round(self.fscores,4)})

def dataio_prep(hparams):
    @sb.utils.data_pipeline.takes("Show","EpId", "ClipId")
    @sb.utils.data_pipeline.provides("id", "waveform")
    def audio_pipeline(Show, EpId, ClipId):
        EpId = int(EpId)
        file = f"{hparams['data_folder']}/sep28k_clips/{Show}/{EpId}/{Show}_{EpId}_{ClipId}.wav"
        waveform, _ = torchaudio.load(file, normalize=True)
        audio = waveform
        return (EpId, int(ClipId)), audio.squeeze()
    @sb.utils.data_pipeline.takes("Prolongation", "Block", "SoundRep", "WordRep", "Interjection", "NoStutteredWords")
    @sb.utils.data_pipeline.provides("label", "unsure")
    def get_label(p, b, sr, wr, inter, f):
        label, unsure = get_labels(p,b,sr,wr,inter,f)
        return label, unsure

    datasets={}
    for dataset in ["train", "valid", "test"]:
        print(f"----------- Processing {dataset} ------------------------")
        csv_path=f'{hparams["data_folder"]}/SEP28k-E_{dataset}.csv'
        datasets[f"{dataset}"] = sb.dataio.dataset.DynamicItemDataset.from_csv(
                csv_path=csv_path,
                dynamic_items=[audio_pipeline, get_label],
                output_keys=["id", "waveform", "label", "unsure"],
            )
        
        if(hparams["remove_unsure"]):
            counter_u =0
            for i in range(len(datasets[dataset])):
                if(datasets[dataset][i]["unsure"]==1):
                    counter_u +=1
            d = datasets[dataset].filtered_sorted(sort_key="unsure", reverse=True, select_n=len(datasets[dataset])-counter_u)
            datasets[dataset] = d
        print(len(datasets[dataset]))
    return datasets

def get_labels(p,b,sr,wr,inter,f):
    label = torch.tensor([int(p),int(b),int(sr),int(wr),int(inter)])
    classes = torch.tensor([hparams["Prolongation"], hparams["Block"], hparams["SoundRep"],hparams["WordRep"],hparams["Interjection"]])
    label = label * classes
    out = torch.any(label>=hparams["annot_value"])
    if int(f) < hparams["annot_value"] and out == False:
        unsure = 1 #handle "unsure" samples
    else:
        unsure = 0
    return torch.tensor([int(out)]), torch.tensor([unsure])

from sep28k_prepare import prepare_sep28k

if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    with hp.hyperparameter_optimization(objective_key="loss") as hp_ctx: # <-- Initialize the context
        hparams_file, run_opts, overrides = hp_ctx.parse_arguments(sys.argv[1:]) # <-- Replace sb with hp_ctx
        with open(hparams_file) as fin:
            hparams = load_hyperpyyaml(fin, overrides)
            try:
                hparams["output_folder"] = hparams["output_folder"]+hp.get_trial_id()
                writer = SummaryWriter("/tensorboard")
            except:
                writer = SummaryWriter(hparams["output_folder"]+"/tensorboard")

        prepare_sep28k(hparams["data_folder"])
        # Create experiment directory
        sb.create_experiment_directory(
            experiment_directory=hparams["output_folder"],
            hyperparams_to_save=hparams_file,
            overrides=overrides,
        )
        datasets = dataio_prep(hparams)
        hparams["dataloader_opts"]["collate_fn"] = PaddedBatch
        # Initialize trainer
        opt_class = partial(hparams["opt_class"].func, lr=float(hparams["opt_class"].keywords["lr"]))
        detect_brain = SEP28kBrain(
            modules= hparams["modules"],
            opt_class=opt_class,
            run_opts=run_opts,
            hparams=hparams,
            checkpointer=hparams["checkpointer"],
        )
        detect_brain.best_loss = 100000
        # Fit dataset
        detect_brain.fit(
            epoch_counter=hparams["counter"],
            train_set=datasets[f"train"],
            valid_set=datasets[f"valid"],
            train_loader_kwargs=hparams["dataloader_opts"],
            valid_loader_kwargs=hparams["dataloader_opts"],
        )
        
        print("*"*20, "Evaluation", "*"*20)
        detect_brain.evaluate(
            datasets[f"test"],
            test_loader_kwargs=hparams["dataloader_opts"],
        )

        writer.add_hparams(overrides,{'score/F1-macro': detect_brain.test_fscore,})
        
        hp.report_result(detect_brain.results)
        writer.flush()
        writer.close()
        #detect_brain.checkpointer.delete_checkpoints(num_to_keep=0)
