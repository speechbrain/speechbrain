import logging
from speechbrain.utils.edit_distance import wer_summary

logger = logging.getLogger(__name__)


class TrainLogger:
    def __init__(self, summary_fns):
        self.summary_fns = summary_fns
        self.stats = {}

    def add_batch(self, stats, phase):
        if phase not in self.stats:
            self.stats[phase] = {}

        for key in stats:
            if key not in self.stats[phase]:
                self.stats[phase][key] = []
            if isinstance(stats[key], list):
                self.stats[phase][key].extend(stats[key])
            else:
                self.stats[phase][key].append(stats[key])

    def summarize(self, epoch_stats):
        log_string = " - ".join(f"{k} {v}" for k, v in epoch_stats.items())
        self.summary = {}
        for phase in self.stats:
            self.summary[phase] = {}
            for key in self.stats[phase]:
                value = self.summary_fns[key](self.stats[phase][key])
                self.summary[phase][key] = value
                log_string += f" - {phase} {key}: {value:.02f}"
        self.stats = {}

        return log_string

    def log_epoch(self, epoch_stats):
        logger.info(self.summarize(epoch_stats))


class TextLogger(TrainLogger):
    def __init__(self, summary_fns, save_file):
        self.summary_fns = summary_fns
        self.save_file = save_file
        self.stats = {}

    def log_epoch(self, epoch_stats, verbose=True):
        summary = self.summarize(epoch_stats)
        with open(self.save_file, "a") as fout:
            print(summary, file=fout)
        if verbose:
            logger.info(summary)


def float_summary(float_list):
    return float(sum(float_list) / len(float_list))


def error_rate_summary(error_rate_list):
    summary = wer_summary(error_rate_list)
    return summary["WER"]
