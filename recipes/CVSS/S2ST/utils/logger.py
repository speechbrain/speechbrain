import wandb
from speechbrain.utils.train_logger import TrainLogger

class WandBLogger(TrainLogger):
    """Logger for wandb. To be used the same way as TrainLogger. Handles nested dicts as well.
    An example on how to use this can be found in recipes/Voicebank/MTL/CoopNet/"""

    def __init__(self, *args, **kwargs):
        try:
            self.run = kwargs.pop("initializer", None)(
                *args, **kwargs) #, config=config_dict)
        except Exception as e:
            raise e("There was an issue with the WandB Logger initialization")

    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=False,
    ):
        """See TrainLogger.log_stats()"""

        logs = {}
        for dataset, stats in [
            ("train", train_stats),
            ("valid", valid_stats),
            ("test", test_stats),
        ]:
            if stats is None:
                continue
            logs[dataset] = stats

        step = stats_meta.get("epoch", None)
        if step is not None:  # Useful for continuing runs that crashed
            self.run.log({**logs, **stats_meta}, step=step)
        else:
            self.run.log({**logs, **stats_meta})

    def log_audio(
        self,
        name,
        audio,
        sample_rate,
        caption=None,
        step=None
    ):
        self.run.log({name: wandb.Audio(audio, caption=caption, sample_rate=sample_rate)}, step=step)

    def log_figure(self, name, image, caption=None, step=None):
        self.run.log({name: wandb.Image(image, caption=caption,)}, step=step)