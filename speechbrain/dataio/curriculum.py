"""
Utilities for curriculum learning

Authors
* Artem Ploujnikov 2022
"""
import torch
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset, FilteredSortedDynamicItemDataset


SAMPLE_OUTPUTS = [
    "wrd_count",
    "sig",
    "wrd_start",
    "wrd_end",
    "phn_start",
    "phn_end",
    "wrd",
    "char",
    "phn"
]


class CurriculumSpeechDataset(DynamicItemDataset):
    """A derivative dynamic dataset that allows to perform
    curriculum learning over a speech dataset with phoneme
    alignments similar to LibriSpeech-Alignments. The dataset
    selects sub-samples within the specified length range in words
    
    Arguments
    ---------
    from_dataset: DynamicItemDataset
        a base dataset compatible with alignments-enhanced LibriSpeech
    min_words: int
        the minimum number of words to sample from each dataset item
    max_words: int
        the maximum number of words to sample from each dataset item
    num_samples: int
        the number of samples per epoch
    sample_rate: int
        the audio sampling rate, in Hertz
    """
    def __init__(
        self,
        from_dataset,
        min_words=1,
        max_words=3,
        num_samples=None,
        sample_rate=16000,
    ):
        super().__init__(data=from_dataset.data)
        self.base_dataset = from_dataset
        self.min_words = min_words
        self.max_words = max_words
        self.num_samples = num_samples
        self.sample_rate = sample_rate
        self.data_id_indices = {
            data_id: idx 
            for idx, data_id in enumerate(self.data_ids)}
        self.sample_segments(self.base_dataset)
        self.setup_pipeline()
        self.pipeline = PipelineWrapper(self.pipeline, SAMPLE_OUTPUTS)

    def sample_segments(self, dataset):
        """Samples parts of the audio file at specific word boundaries
        
        Arguments
        ---------
        datset: DynamicItemDataset
            the dataset from which to sample
        """
        # Exclude samples than have fewer
        # words than the minimum
        dataset = dataset.filtered_sorted(
            key_min_value={
                "wrd_count": self.min_words
            }                
        )
        keys = ["wrd_count", "wrd_start", "wrd_end"]
        with dataset.output_keys_as(keys):            
            wrd_count = torch.tensor(self._pluck("wrd_count"))
            wrd_start = self._pluck("wrd_start")
            wrd_end = self._pluck("wrd_end")
            

        # Randomly sample word counts in the 
        # range form num_words to last_words
        self.sample_word_counts = torch.randint(
            low=self.min_words,
            high=self.max_words + 1,
            size=(len(dataset),)
        )

        # Sample relative offsets, from 0.0 to 1.0.
        # 0.0 corresponds to the beginning of the
        # utterance, where as 1.0 represents wrd_count - n
        # where n is the sampled word count
        sample_offsets_rel = torch.rand(len(dataset))
        
        # Determine the maximum possible offsets
        max_offset = wrd_count - self.sample_word_counts
        self.wrd_offset_start = (sample_offsets_rel * max_offset).floor().int()
        self.wrd_offset_end = self.wrd_offset_start + self.sample_word_counts
        sample_start = torch.tensor([
            item[idx]
            for item, idx in zip(wrd_start, self.wrd_offset_start)
        ])
        sample_end = torch.tensor([
            item[idx - 1]
            for item, idx in zip(wrd_end, self.wrd_offset_end)
        ])
        sample_start_idx = time_to_index(
            sample_start, self.sample_rate)
        sample_end_idx = time_to_index(
            sample_end, self.sample_rate
        )
        self.sample_start_idx = sample_start_idx
        self.sample_end_idx = sample_end_idx

    def _pluck(self, key):
        return [self.data[data_id][key] for data_id in self.data_ids]

    def setup_pipeline(self):
        @sb.utils.data_pipeline.takes(
            "id",
            "wav",
            "phn_start",
            "phn_end",
            "wrd_start",
            "wrd_end",
            "wrd",
            "phn"
        )
        @sb.utils.data_pipeline.provides(
            "_wrd_count",
            "_sig",
            "_wrd_start",
            "_wrd_end",
            "_phn_start",
            "_phn_end",
            "_wrd",
            "_char",
            "_phn"
        )
        def cut_sample(
            data_id,
            wav,
            wrd_start,
            wrd_end,
            phn_start,
            phn_end,
            wrd,
            phn
        ):
            idx = self.data_id_indices[data_id]
            # wrd_count
            yield self.sample_word_counts[idx]
            sample_start_idx = self.sample_start_idx[idx]
            sample_end_idx = self.sample_end_idx[idx]
            sig = sb.dataio.dataio.read_audio(wav)
            sig = sig[sample_start_idx:sample_end_idx]
            # sig
            yield sig
            wrd_offset_start = self.wrd_offset_start[idx]
            wrd_offset_end = self.wrd_offset_end[idx]
            # wrd_start
            yield cut_offsets(wrd_start, wrd_offset_start, wrd_offset_end)
            # wrd_end
            yield cut_offsets(wrd_end, wrd_offset_start, wrd_offset_end)
            # phn_start
            phn_start, phn_from, phn_to = cut_offsets_rel(
                wrd_start, phn_start, wrd_offset_start, wrd_offset_end)
            yield phn_start
            # phn_end
            phn_end, _, _ = cut_offsets_rel(
                wrd_end, phn_end, wrd_offset_start, wrd_offset_end)
            yield phn_end
            # wrd
            wrd_sample = wrd[wrd_offset_start:wrd_offset_end]
            yield wrd_sample
            yield " ".join(wrd_sample).upper()
            phn = phn[phn_from: phn_to]
            yield phn
        
        self.add_dynamic_item(cut_sample)

    def sample(self):
        """Retrieves a sample of the based dataset"""
        sample_data_ids = torch.tensor(self.base_dataset.data_ids)
        if self.num_samples:
            sample_indexes = torch.multinomial(
                num_samples=self.num_samples,
                replacement=self.num_samples > len(self.base_dataset)
            )
            sample_data_ids = sample_data_ids[sample_indexes]

        return FilteredSortedDynamicItemDataset(
            from_dataset=self.base_dataset,
            data_ids=sample_data_ids
        )


class PipelineWrapper:
    def __init__(self, pipeline, replace_keys):
        self.pipeline = pipeline
        self.key_map = {
            key: f"_{key}"
            for key in replace_keys
        }

    def compute_outputs(self, data):
        result = self.pipeline.compute_outputs(data)
        for key, key_r in self.key_map.items():
            if key_r in result:
                result[key] = result[key_r]
                del result[key_r]
        return result

    def set_output_keys(self, keys):
        keys_r = {self.key_map.get(key, key) for key in keys}
        self.pipeline.set_output_keys(keys_r)


def time_to_index(times, sample_rate):
    """Converts a collection of time values to a list of
    wave array indexes at the specified sample rate
    
    Arguments
    ---------
    times: enumerable
        a list of time values
    sample_rate: int
        the sample rate (in hertz)

    Returns
    -------
    result: list
        a collection of indexes
    """

    if not torch.is_tensor(times):
        times = torch.tensor(times)
    
    return (
        (times * sample_rate)
        .floor()
        .int()
        .tolist()
    )

def cut_offsets(offsets, start, end):
    """Given an array of offsets (e.g. word start times),
    returns a segment of it from <start> to <end> re-computed
    to begin at 0

    Arguments
    ---------
    offsets: list|torch.tensor
        a list or tensor of offsets
    
    start: int
        the starting index
    
    end: int
        the final index

    Returns
    -------
    result: list
        the re-calculated offset list
    """
    segment = offsets[start:end]
    if not torch.is_tensor(segment):
        segment = torch.tensor(segment)
    return (segment - segment[0]).tolist()

def cut_offsets_rel(
    offsets,
    ref_offsets,
    start,
    end
):
    """Given a sequence of offsets (e.g. phoneme offsets)
    and a reference sequence (e.g. sequence of words), finds
    the range in <offsets> corresponding to the specified range
    in <ref_offsets>

    Arguments
    ---------
    offsets: list|torch.Tensor
        a collection of offsets

    ref_offsets: list|torch.Tensor
        reference offsets

    Returns
    -------
    result: list
        the corresponding values in offsets
    start: int
        the start index
    end: int
        the end index
    """
    if not torch.is_tensor(offsets):
        offsets = torch.tensor(offsets)
    if not torch.is_tensor(ref_offsets):
        ref_offsets = torch.tensor(ref_offsets)
    start_value = ref_offsets[start].item()
    end_value = ref_offsets[end].item()
    condition = (
        (offsets >= start_value)
        &
        (offsets < end_value)
    )
    result = offsets[condition]
    result -= result[0].item()
    idx = condition.nonzero()
    return result.tolist(), idx.min().item(), idx.max().item() + 1
