#!/usr/bin/env python3
"""
Classes to perform dialogue state evaluation outside of training or during training with a running tracker.

Author
    * Lucas Druart 2024
"""

import argparse
from tqdm import tqdm
import numpy as np
import json
import os

multiwoz_slot_types = [
    "attraction-area",
    "attraction-name",
    "attraction-type",
    "hotel-area",
    "hotel-name",
    "hotel-type",
    "hotel-day",
    "hotel-people",
    "hotel-pricerange",
    "hotel-stay",
    "hotel-stars",
    "hotel-internet",
    "hotel-parking",
    "restaurant-area",
    "restaurant-name",
    "restaurant-food",
    "restaurant-day",
    "restaurant-people",
    "restaurant-pricerange",
    "restaurant-time",
    "taxi-arriveby",
    "taxi-departure",
    "taxi-destination",
    "taxi-leaveat",
    "train-arriveby",
    "train-departure",
    "train-destination",
    "train-leaveat",
    "train-people",
    "train-day",
    "hospital-department",
    "bus-people",
    "bus-leaveat",
    "bus-arriveby",
    "bus-day",
    "bus-destination",
    "bus-departure",
]

multiwoz_time_slots = [
    "restaurant-time",
    "taxi-arriveby",
    "taxi-leaveat",
    "train-arriveby",
    "train-leaveat",
]

multiwoz_open_slots = [
    "attraction-name",
    "hotel-name",
    "restaurant-name",
    "taxi-departure",
    "taxi-destination",
    "train-departure",
    "train-destination",
]


def dialogueState_str2dict(
    dialogue_state: str, slot_type_filtering: list[str] = None
):
    """
    Converts the ; separated Dialogue State linearization to a domain-slot-value dictionary.
    When *slot_type_filtering* is provided, it filters out the slots which are not part of this list.
    """
    dict_state = {}

    # We consider every word after "[State] " to discard the transcription if present.
    dialogue_state = dialogue_state.split("[State] ")[-1]
    if ";" not in dialogue_state:
        return {}
    else:
        slots = dialogue_state.split(";")
        for slot_value in slots:
            if "=" not in slot_value:
                continue
            else:
                slot, value = (
                    slot_value.split("=")[0].strip(),
                    slot_value.split("=")[1].strip(),
                )
                if slot_type_filtering and slot not in slot_type_filtering:
                    continue
                elif "-" in slot:
                    domain, slot_type = (
                        slot.split("-")[0].strip(),
                        slot.split("-")[1].strip(),
                    )
                    if domain not in dict_state.keys():
                        dict_state[domain] = {}
                    dict_state[domain][slot_type] = value

        return dict_state


def dialogueState_dict2str(dialogue_state: dict):
    """
    Converts a dialogue state as a dictionary per slot type per domain to a serialized ; separated list
    """
    slots = []
    for domain, slots_values in dialogue_state.items():
        slots.extend(
            [f"{domain}-{slot}={value}" for slot, value in slots_values.items()]
        )

    return "; ".join(slots)


class JGATracker:
    """
    Class to track the Joint-Goal Accuracy during training. Keeps track of the number of correct and total dialogue states.
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def append(self, predictions: list[str], targets: list[str]):
        """
        This function is for updating the stats according to the a batch of predictions and targets.

        Arguments
        ----------
        predictions : list[str]
            Predicted dialogue states.
        targets : list[str]
            Target dialogue states.
        """
        for prediction, reference in zip(predictions, targets):
            pred = dialogueState_str2dict(prediction)
            ref = dialogueState_str2dict(reference)
            if pred == ref:
                self.correct += 1
            self.total += 1

    def summarize(self):
        """
        Averages the current Joint-Goal Accuracy (JGA).
        """
        return (
            round(100 * self.correct / self.total, 2)
            if self.total != 0
            else 100.00
        )


class DSTMetrics:
    """
    Class to compute the Dialogue State Tracking metrics i.e. Joint-Goal Accuracy and slot F1.

    Arguments
    ----------
    slot_types: List
        List of slot types to consider for slot F1 and slot prediction filtering.
    time_slots: List
        List of time related slot types to compute average F1 per time slot group.
    open_slots: List
        List of open value slot types to compute average F1 per open value slot group.
    """

    def __init__(
        self,
        slot_types=multiwoz_slot_types,
        time_slots=multiwoz_time_slots,
        open_slots=multiwoz_open_slots,
    ):
        """
        Prepares the class for data.
        """
        self.references = {}
        self.predictions = {}
        self.slot_types = slot_types
        self.time_slots = time_slots
        self.open_slots = open_slots

    def add_prediction(
        self, prediction: str, dialogue_id: str, turn_id: str, filtering=True
    ):
        """
        Add the provided prediction to the set of predictions grouped per dialogue.
        A filtering can be applied ot discard slots not part of the defined slot types.
        """
        if dialogue_id not in self.predictions:
            self.predictions[dialogue_id] = {}
        if filtering:
            self.predictions[dialogue_id][turn_id] = dialogueState_str2dict(
                prediction, slot_type_filtering=self.slot_types
            )
        else:
            self.predictions[dialogue_id][turn_id] = dialogueState_str2dict(
                prediction
            )

    def add_reference(self, reference: str, dialogue_id: str, turn_id: str):
        """
        Add the provided reference to the set of references grouped per dialogue.
        """
        if dialogue_id not in self.references:
            self.references[dialogue_id] = {}
        self.references[dialogue_id][turn_id] = dialogueState_str2dict(
            reference
        )

    def bootstrap_dialogues(self):
        """
        Samples, with replacement, N=size_of_dataset dialogues from the set of examples.
        """
        nbr_dialogues = len(self.dialogues)
        sample = np.random.choice(self.dialogues, nbr_dialogues)

        return sample

    def prepare_samples(
        self, evaluate_ci=False, number_bootstrap_samples=1000, alpha=5
    ):
        """
        Prepares the samples from the dataset to compute the metrics and the JGA confidence intervals with bootstrapping.
        """
        self.dialogues = np.asarray([k for k in self.references.keys()])
        self.evaluate_ci = evaluate_ci
        if self.evaluate_ci:
            self.number_of_samples = number_bootstrap_samples
            self.samples = [
                self.bootstrap_dialogues()
                for _ in range(self.number_of_samples)
            ]
            # Last sample is the dataset itself for metric computation
            self.samples.append(self.dialogues)
        else:
            self.number_of_samples = 0
            self.samples = [self.dialogues]
        self.alpha = alpha

    def slot_precision_recall(self):
        """
        Computes for each slot:
            - True Positives (the slot should have been predicted and was predicted)
            - False Positives (the slot should not have been predicted but was predicted)
            - False Negative (the slot should have been predicted but was not predicted)
        For each slot, we consider just the slot type and the slot type together with its value.
        """

        self.slot_scores = {}
        self.slot_value_scores = {}

        print("Computing Slot Precision Scores...")
        # For slot precision and recall, we only compute the metric over the whole dataset without evaluating the CIs
        k = self.number_of_samples
        sampled_dialogues = self.dialogues

        self.slot_scores[f"sample {k}"] = {
            slot_name: {
                "true-positive": 0,
                "false-positive": 0,
                "false-negative": 0,
            }
            for slot_name in self.slot_types
        }
        self.slot_value_scores[f"sample {k}"] = {
            slot_name: {
                "true-positive": 0,
                "false-positive": 0,
                "false-negative": 0,
            }
            for slot_name in self.slot_types
        }
        for dialogue_id in sampled_dialogues:
            for dialogue_turn, reference_state in self.references[
                dialogue_id
            ].items():
                prediction = self.predictions[dialogue_id][dialogue_turn]

                for domain, slots in reference_state.items():
                    for slot, value in slots.items():
                        if domain in prediction:
                            if slot in prediction[domain]:
                                self.slot_scores[f"sample {k}"][
                                    f"{domain}-{slot}"
                                ]["true-positive"] += 1
                                if value == prediction[domain][slot]:
                                    self.slot_value_scores[f"sample {k}"][
                                        f"{domain}-{slot}"
                                    ]["true-positive"] += 1
                                else:
                                    self.slot_value_scores[f"sample {k}"][
                                        f"{domain}-{slot}"
                                    ]["false-negative"] += 1
                            else:
                                self.slot_scores[f"sample {k}"][
                                    f"{domain}-{slot}"
                                ]["false-negative"] += 1
                        else:
                            self.slot_scores[f"sample {k}"][f"{domain}-{slot}"][
                                "false-negative"
                            ] += 1

            # Counting the false positives
            for dialogue_turn, predicted_state in self.predictions[
                dialogue_id
            ].items():
                for domain, slots in predicted_state.items():
                    for slot, value in slots.items():
                        if (
                            f"{domain}-{slot}"
                            not in self.slot_scores[f"sample {k}"]
                        ):
                            # slots which do not exist are ignored for scores
                            continue
                        elif (
                            domain
                            in self.references[dialogue_id][dialogue_turn]
                        ):
                            if (
                                slot
                                not in self.references[dialogue_id][
                                    dialogue_turn
                                ][domain]
                            ):
                                self.slot_scores[f"sample {k}"][
                                    f"{domain}-{slot}"
                                ]["false-positive"] += 1
                                self.slot_value_scores[f"sample {k}"][
                                    f"{domain}-{slot}"
                                ]["false-positive"] += 1
                        else:
                            self.slot_scores[f"sample {k}"][f"{domain}-{slot}"][
                                "false-positive"
                            ] += 1

    def jga(self):
        """
        Computes the per-turn Joint-Goal Accuracy (JGA) for all data samples.
        """

        self.jga_turn_scores = {}
        self.jga_scores = {}
        print("Computing Joint-Goal Accuracy...")
        for k, sampled_dialogues in enumerate(tqdm(self.samples)):
            # Joint-Goal Accuracy is computed per turn and (if needed) averaged over all turns.
            self.jga_turn_scores[f"sample {k}"] = {}

            for dialog_id in sampled_dialogues:
                for dialogue_turn, reference in self.references[
                    dialog_id
                ].items():
                    prediction = self.predictions[dialog_id][dialogue_turn]

                    if dialogue_turn not in self.jga_turn_scores[f"sample {k}"]:
                        self.jga_turn_scores[f"sample {k}"][dialogue_turn] = {
                            "correct": 0,
                            "total": 0,
                        }
                    if prediction == reference:
                        self.jga_turn_scores[f"sample {k}"][dialogue_turn][
                            "correct"
                        ] += 1
                    self.jga_turn_scores[f"sample {k}"][dialogue_turn][
                        "total"
                    ] += 1

            total_correct = sum(
                [
                    turn["correct"]
                    for _, turn in self.jga_turn_scores[f"sample {k}"].items()
                ]
            )
            total = sum(
                [
                    turn["total"]
                    for _, turn in self.jga_turn_scores[f"sample {k}"].items()
                ]
            )
            self.jga_scores[f"sample {k}"] = round(
                100 * total_correct / total, 1
            )

    def get_average_jga(self):
        round(100 * self.correct / self.total, 2) if self.total != 0 else 100.00

    def read_multiwoz_files(
        self, predictions_file: str, reference_manifest: str, filtering: bool
    ):
        self.file = predictions_file
        print("\nExtracting the references...\n")
        with open(reference_manifest, "r") as references:
            dialog_id = ""
            for line in tqdm(references):
                if line.__contains__("END_OF_DIALOG"):
                    pass
                else:
                    fields = line.split(" ", 7)
                    # A line looks like: line_nr: [N] dialog_id: [D.json] turn_id: [T] text: (user:|agent:) [ABC] state: domain1-slot1=value1; domain2-slot2=value2
                    key_map = {
                        "line_nr": 1,
                        "dialog_id": 3,
                        "turn_id": 5,
                        "text": 7,
                    }
                    if (
                        fields[key_map["dialog_id"]].split(".json")[0]
                        != dialog_id
                    ):
                        # Arriving on a new dialog we reset our dialogue_id
                        dialog_id = fields[key_map["dialog_id"]].split(".json")[
                            0
                        ]
                    turn_id = fields[key_map["turn_id"]]

                    # User turn line
                    if int(turn_id) % 2 == 1:
                        # Extracting the text part (transcription and state) of the line
                        text_split = fields[key_map["text"]].split("state:")
                        state = text_split[-1].strip()
                        self.add_reference(
                            state,
                            dialogue_id=dialog_id,
                            turn_id=f"Turn-{turn_id}",
                        )

        print("\nExtracting the predictions...\n")
        with open(predictions_file, "r") as predictions:
            for line in tqdm(predictions):
                # The predictions csv is composed of the id and the prediction
                fields = line.split(",", 1)
                dialogue_id = fields[0].split("/")[-2]
                turn_id = fields[0].split("/")[-1]
                self.add_prediction(
                    fields[1].strip(),
                    dialogue_id=dialogue_id,
                    turn_id=turn_id,
                    filtering=filtering,
                )

        if self.references.keys() != self.predictions.keys():
            raise AssertionError(
                f"Careful the predictions ({predictions_file}) and references ({reference_manifest}) do not concern strictly the same set of examples."
            )

    def read_spokenwoz_files(self, predictions_file: str, reference_manifest: str, filtering: bool):

        with open(reference_manifest, "r") as data:
            for line in data:
                annotations = json.loads(line)
        
        if "dev" in reference_manifest:
            # Selecting only dialogues for dev set
            dev_ids = []
            folder = os.path.dirname(reference_manifest)
            with open(os.path.join(folder, "valListFile.json"), "r") as val_list_file:
                for line in val_list_file:
                    dev_ids.append(line.strip())
            dialogues_to_remove = [k for k, _ in annotations.items() if k not in dev_ids]
            for dialog_id in dialogues_to_remove:
                del annotations[dialog_id]
        
        for dialog_id, dialog_info in annotations.items():
            for turn_id, turn_info in enumerate(dialog_info["log"]):
                if turn_id % 2 == 1:
                    # Dialogue States annotations are on Agent turns 
                    state = []
                    for domain, info in turn_info["metadata"].items():
                        for slot, value in info["book"].items():
                            if slot != "booked" and value != '':
                                state.append(f'{domain}-{slot}={value}')
                        for slot, value in info["semi"].items():
                            if value != "":
                                # One example in train set has , between numbers
                                state.append(f'{domain}-{slot}={value.replace(",", "")}')
                    self.add_reference("; ".join(state), dialogue_id=dialog_id, turn_id=f'Turn-{turn_id-1}')
        
        print("\nExtracting the predictions...\n")
        with open(predictions_file, "r") as predictions:
            for line in tqdm(predictions):
                # The predictions csv is composed of the id and the prediction
                fields = line.split(',', 1)
                # Example of id: SNG1751_Turn-26
                dialogue_id = fields[0].split('/')[-1].split('_')[0]
                turn_id = fields[0].split('/')[-1].split('_')[1]
                self.add_prediction(fields[1].strip(), dialogue_id=dialogue_id, turn_id=turn_id, filtering=filtering)

        if self.references.keys() != self.predictions.keys():
            raise AssertionError(f"Careful the predictions ({predictions_file}) and references ({reference_manifest}) do not concern strictly the same set of examples.")

    def summary(self):
        self.jga()
        self.slot_precision_recall()
        summary = f"==================Metric report of {self.file}==================\n"

        evaluated_jga = self.jga_scores[f"sample {self.number_of_samples}"]
        if self.evaluate_ci:
            # Get confidence interval over values
            # https://github.com/luferrer/ConfidenceIntervals/blob/main/confidence_intervals/confidence_intervals.py#L165
            jga_values = [
                jga
                for sample, jga in self.jga_scores.items()
                if sample != f"sample {self.number_of_samples}"
            ]
            jga_low = np.percentile(jga_values, self.alpha / 2)
            jga_high = np.percentile(jga_values, 100 - self.alpha / 2)
            summary += f"Joint-Goal Accuracy = {evaluated_jga}% ({jga_low}, {jga_high})\n"
        else:
            summary += f"Joint-Goal Accuracy = {evaluated_jga}%\n"

        evaluated_per_turn_jga = {
            turn: round(100 * values["correct"] / values["total"], 1)
            for turn, values in self.jga_turn_scores[
                f"sample {self.number_of_samples}"
            ].items()
        }
        if self.evaluate_ci:
            per_turn_jga_values = {
                turn: []
                for turn in self.jga_turn_scores[
                    f"sample {self.number_of_samples}"
                ].keys()
            }
            for sample, per_turn_jga in self.jga_turn_scores.items():
                if sample != f"sample {self.number_of_samples}":
                    for turn, stats in per_turn_jga.items():
                        per_turn_jga_values[turn].append(
                            100 * stats["correct"] / stats["total"]
                        )
            low_per_turn_jga = {
                turn: np.percentile(values, self.alpha / 2)
                for turn, values in per_turn_jga_values.items()
            }
            high_per_turn_jga = {
                turn: np.percentile(values, 100 - self.alpha / 2)
                for turn, values in per_turn_jga_values.items()
            }
            turn_cis = [
                (value, low_per_turn_jga[turn], high_per_turn_jga[turn])
                for turn, value in evaluated_per_turn_jga.items()
            ]
            summary += "\tPer-turn: \n\t\t" + f"{turn_cis}\n\n"
        else:
            summary += (
                "\tPer-turn: \n\t\t"
                + f"{[value for value in evaluated_per_turn_jga.values()]}\n\n"
            )

        open_f1s = []
        time_f1s = []
        cat_f1s = []

        summary += "Slot Values Scores:\n"
        for slot, scores in self.slot_value_scores[
            f"sample {self.number_of_samples}"
        ].items():
            tp = scores["true-positive"]
            fp = scores["false-positive"]
            fn = scores["false-negative"]
            precision = tp / (tp + fp) if (tp + fp) != 0 else 1
            recall = tp / (tp + fn) if (tp + fn) != 0 else 1
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) != 0
                else 1
            )
            if slot in self.open_slots:
                open_f1s.append(f1)
            elif slot in self.time_slots:
                time_f1s.append(f1)
            else:
                cat_f1s.append(f1)

        summary += f"\t- Open slots:\n"
        summary += f"\t\t- F1s: {open_f1s}\n"
        summary += f"\t- Time slots:\n"
        summary += f"\t\t- F1s: {time_f1s}\n"
        summary += f"\t- Categorical slots:\n"
        summary += f"\t\t- F1s: {cat_f1s}\n"

        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference_manifest",
        type=str,
        help="The path to the reference txt file.",
        default="../data/dev_manifest.txt",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        help="The path where to find the csv file with the models predictions.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="multiwoz",
        help='Dataset ("multiwoz" or "spokenwoz") for which the evaluation is done. Selects the way to read the reference file and the slot types.'
    )
    parser.add_argument(
        "--no_filtering",
        action="store_true",
        default=False,
        help="Deactivates the slot ontology predictions filtering.",
    )
    parser.add_argument(
        "--evaluate_ci",
        action="store_true",
        default=False,
        help="Whether to evaluate the confidence intervals of the JGA.",
    )
    args = parser.parse_args()

    if args.dataset == "multiwoz":
        metrics = DSTMetrics()
        metrics.read_multiwoz_files(
            predictions_file=args.predictions,
            reference_manifest=args.reference_manifest,
            filtering=not args.no_filtering,
        )
    elif args.dataset == "spokenwoz":
        metrics = DSTMetrics()
        # Adding the extended slot types specific to spokenwoz
        metrics.slot_types += ['profile-name', 'profile-phonenumber', 'profile-idnumber', 'profile-email', 'profile-platenumber']
        metrics.read_spokenwoz_files(
            predictions_file=args.predictions, 
            reference_manifest=args.reference_manifest, 
            filtering=not args.no_filtering
            )
    else:
        parser.error('Argument dataset should be either "multiwoz" or "spokenwoz".')
    
    metrics.prepare_samples(evaluate_ci=args.evaluate_ci)

    print(metrics.summary())


if __name__ == "__main__":
    main()
