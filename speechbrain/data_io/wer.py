import sys
import speechbrain.utils.edit_distance as edit_distance
from speechbrain.data_io.data_io import create_dataloader
from speechbrain.utils.logger import logger_write
from speechbrain.utils.input_validation import (check_opts,
                                                check_inputs,
                                                check_input_shapes)

## These internal functions convert CSV data dictionaries to 
## generators and dicts which simply provide the info in
## one field of the csv. Here used because we just want to
## access the label sequences.
def _label_sequence_generator(data_dict, field):
    for key in data_dict['data_list']:
        yield key, data_dict[key][field]['data'].split()

def _label_sequence_dict(data_dict, field):
    return {key: data_dict[key][field]['data'].split() for key in data_dict['data_list']}

class ComputeAndSaveWERAndAlignments:
    """
    Description:
        Takes in two CSV filenames, computes WER and Alignments
        based on specified CSV fields,
        then writes the WER and alignments in a file.
    Input(init):
        ref_csv (type: string, mandatory) Filepath to reference label CSV
        ref_csv_field (type: string, mandatory) Field to read in reference CSV
        hyp_csv (type: string, mandatory) Filepath to hypothesis label CSV
        hyp_csv_field (type: string, mandatory) Field to read in hypothesis CSV
        outfile (type: string, optional) File to write output in. If not 
            specified, outfile becomes <output_folder>/wer.txt 
            (<output_folder> from global config)
        scoring_mode (type: string, optional)
            Must be one of 'strict', 'present', 'all':
            'strict': raise error for missing hypothesis
            'present': only score existing hypotheses
            'all': score missing hypotheses as empty
    Input(call): <empty input list>
    Output(call): <None> (Output is written to <outfile>)
    Author:)
        Aku Rouhe 2020
    """
   
    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        functions=None,
        logger=None,
        first_input=None
        ):

        # Logger setup
        self.logger = logger

        # Here are summarized the expected options for this class
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "ref_csv": ("str", "mandatory"),
            "ref_csv_field": ("str", "mandatory"),
            "hyp_csv": ("str", "mandatory"),
            "hyp_csv_field": ("str", "mandatory"),
            "outfile": ("str", "optional", ""),
            "scoring_mode": ("one_of(strict,present,all)", "optional", "strict")
        }

        # Check, cast , and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )
        if not self.conf['outfile']:
            #NOTE: "output_folder" is mandatory in root config.
            self.conf['outfile'] = global_config["output_folder"]+ "/wer.txt"

    def __call__(self, input_list):
        ref_data_loader = create_dataloader({
            'class_name': 'core.loop',
            'csv_file': self.conf['ref_csv'],
            })
        hyp_data_loader = create_dataloader({
            'class_name': 'core.loop',
            'csv_file': self.conf['hyp_csv'],
            })
        ref_data_dict = ref_data_loader.generate_data_dict()
        hyp_data_dict = hyp_data_loader.generate_data_dict()
        ref_reader = _label_sequence_generator(ref_data_dict, 
                self.conf['ref_csv_field'])
        hyp_dict = _label_sequence_dict(hyp_data_dict, 
                self.conf['hyp_csv_field'])
        details_by_utterance = edit_distance.wer_details_by_utterance(
                ref_reader, hyp_dict, 
                compute_alignments=True, scoring_mode=self.conf['scoring_mode'])
        summary_details = edit_distance.wer_summary(details_by_utterance)
        _print_wer_summary(summary_details)
        with open(self.conf['outfile'], "w") as fo:
            _print_wer_summary(summary_details, file=fo)
            _print_alignments_global_header(file=fo)
            for dets in details_by_utterance:
                if dets["scored"]:
                    _print_alignment_header(dets,file=fo)
                    _print_alignment(
                        dets["alignment"],
                        dets["ref_tokens"],
                        dets["hyp_tokens"],
                        file=fo
                    )


         
# The following internal functions are used to print the computed statistics
# with human readable formatting.
def _print_wer_summary(wer_details, file=sys.stdout):
    # This function essentially mirrors the Kaldi compute-wer output format
    print(
        "%WER {WER:.2f} [ {num_edits} / {num_scored_tokens}, {insertions} ins, {deletions} del, {substitutions} sub ]".format(  # noqa
            **wer_details
        ),
        file=file,
        end="",
    )
    print(
        " [PARTIAL]"
        if wer_details["num_scored_sents"] < wer_details["num_ref_sents"]
        else "",
        file=file,
    )
    print(
        "%SER {SER:.2f} [ {num_erraneous_sents} / {num_scored_sents} ]".format(
            **wer_details
        ),
        file=file,
    )
    print(
        "Scored {num_scored_sents} sentences, {num_absent_sents} not present in hyp.".format(  # noqa
            **wer_details
        ),
        file=file,
    )


def _print_top_wer_utts(top_non_empty, file=sys.stdout):
    print("=" * 80, file=file)
    print("UTTERANCES WITH HIGHEST WER", file=file)
    if top_non_empty:
        print(
            "Non-empty hypotheses -- utterances for which output was produced:",  # noqa
            file=sys.stdout,
        )
        for dets in top_non_empty:
            print("{key} %WER {WER:.2f}".format(**dets), file=file)


def _print_top_wer_spks(spks_by_wer, file=sys.stdout):
    print("=" * 80, file=file)
    print("SPEAKERS WITH HIGHEST WER", file=file)
    for dets in spks_by_wer:
        print("{speaker} %WER {WER:.2f}".format(**dets), file=file)


def _print_alignment(
    alignment, a, b, empty_symbol="<eps>", separator=" ; ", file=sys.stdout
):
    # First, get equal length text for all:
    a_padded = []
    b_padded = []
    ops_padded = []
    for op, i, j in alignment:  # i indexes a, j indexes b
        op_string = str(op)
        a_string = str(a[i]) if i is not None else empty_symbol
        b_string = str(b[j]) if j is not None else empty_symbol
        # NOTE: the padding does not actually compute printed length,
        # but hopefully we can assume that printed length is
        # at most the str len
        pad_length = max(len(op_string), len(a_string), len(b_string))
        a_padded.append(a_string.center(pad_length))
        b_padded.append(b_string.center(pad_length))
        ops_padded.append(op_string.center(pad_length))
    # Then print, in the order Ref, op, Hyp
    print(separator.join(a_padded), file=file)
    print(separator.join(ops_padded), file=file)
    print(separator.join(b_padded), file=file)


def _print_alignments_global_header(
    empty_symbol="<eps>", separator=" ; ", file=sys.stdout
):
    print("=" * 80, file=file)
    print("ALIGNMENTS", file=file)
    print("", file=file)
    print("Format:", file=file)
    print("<utterance-id>, WER DETAILS", file=file)
    # Print the format with the actual
    # print_alignment function, using artificial data:
    a = ["reference", "on", "the", "first", "line"]
    b = ["and", "hypothesis", "on", "the", "third"]
    alignment = [
        (edit_distance.EDIT_SYMBOLS["ins"], None, 0),
        (edit_distance.EDIT_SYMBOLS["sub"], 0, 1),
        (edit_distance.EDIT_SYMBOLS["eq"], 1, 2),
        (edit_distance.EDIT_SYMBOLS["eq"], 2, 3),
        (edit_distance.EDIT_SYMBOLS["sub"], 3, 4),
        (edit_distance.EDIT_SYMBOLS["del"], 4, None),
    ]
    _print_alignment(alignment, a, b, file=file)


def _print_alignment_header(wer_details, file=sys.stdout):
    print("=" * 80, file=file)
    print(
        "{key}, %WER {WER:.2f} [ {num_edits} / {num_ref_tokens}, {insertions} ins, {deletions} del, {substitutions} sub ]".format(  # noqa
            **wer_details
        ),
        file=file,
    )
