#!/usr/bin/env python3
"""
Classes to perform dialogue state evaluation outside of training or during training with a running tracker.

Author
    * Lucas Druart 2024
"""

import json

import numpy as np
from tqdm import tqdm

from speechbrain.utils.metric_stats import dialogue_state_str2dict


class DSTEval:
    """
    Class to evaluate Dialogue State Tracking metrics i.e. Joint-Goal Accuracy and slot's F1.

    Arguments
    ----------
    slot_categories: str
        Path to the json mapping different slots to broader categories depending on their possible values.
        Required for both the full slot list and the slot groups for grouped F1.
    """

    def __init__(self, slot_categories):
        self.references = {}
        self.predictions = {}
        with open(slot_categories, "r") as slots:
            self.slot_categories = json.load(slots)
        self.slot_list = []
        for _, slots in self.slot_categories.items():
            self.slot_list.extend([slot for slot in slots])

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
            self.predictions[dialogue_id][turn_id] = dialogue_state_str2dict(
                prediction, slot_type_filtering=self.slot_list
            )
        else:
            self.predictions[dialogue_id][turn_id] = dialogue_state_str2dict(
                prediction
            )

    def add_reference(self, reference: str, dialogue_id: str, turn_id: str):
        """
        Add the provided reference to the set of references grouped per dialogue.
        """
        if dialogue_id not in self.references:
            self.references[dialogue_id] = {}
        self.references[dialogue_id][turn_id] = dialogue_state_str2dict(
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
            for slot_name in self.slot_list
        }
        self.slot_value_scores[f"sample {k}"] = {
            slot_name: {
                "true-positive": 0,
                "false-positive": 0,
                "false-negative": 0,
            }
            for slot_name in self.slot_list
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

    def read_files(
        self, predictions_file: str, reference_manifest: str, filtering: bool
    ):
        raise NotImplementedError

    def summary(self):
        """
        Computes both turn-level Joint-Goal Accuracy and slot F1.

        Returns
        -------
        summary: str
            Metric report summarizing the computed results.
        """
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

        categorical_f1s = {cat: [] for cat in self.slot_categories.keys()}

        summary += "Slot Values Scores:\n"
        sample_slot_value_scores = self.slot_value_scores[
            f"sample {self.number_of_samples}"
        ]
        for category, slots in self.slot_categories.items():
            for slot in slots:
                scores = sample_slot_value_scores[slot]
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
                categorical_f1s[category].append(f1)

        summary += "\n".join(
            [
                f"\t- {category}:\n \t\t- F1s: {f1s}"
                for category, f1s in categorical_f1s.items()
            ]
        )

        return summary
