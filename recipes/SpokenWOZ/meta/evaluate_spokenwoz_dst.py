import argparse
import json
import os

from tqdm import tqdm

from speechbrain.utils.evaluate_dialogue_state_tracking import DSTEval


class SpokenWOZDSTEval(DSTEval):
    def read_files(
        self, predictions_file: str, reference_manifest: str, filtering: bool
    ):
        self.file = predictions_file
        with open(reference_manifest, "r") as data:
            for line in data:
                annotations = json.loads(line)

        if "dev" in reference_manifest:
            # Selecting only dialogues for dev set
            dev_ids = []
            folder = os.path.dirname(reference_manifest)
            with open(
                os.path.join(folder, "valListFile.json"), "r"
            ) as val_list_file:
                for line in val_list_file:
                    dev_ids.append(line.strip())
            dialogues_to_remove = [
                k for k, _ in annotations.items() if k not in dev_ids
            ]
            for dialog_id in dialogues_to_remove:
                del annotations[dialog_id]

        for dialog_id, dialog_info in annotations.items():
            for turn_id, turn_info in enumerate(dialog_info["log"]):
                if turn_id % 2 == 1:
                    # Dialogue States annotations are on Agent turns
                    state = []
                    for domain, info in turn_info["metadata"].items():
                        for slot, value in info["book"].items():
                            if slot != "booked" and value != "":
                                state.append(f"{domain}-{slot}={value}")
                        for slot, value in info["semi"].items():
                            if value != "":
                                # One example in train set has , between numbers
                                state.append(
                                    f'{domain}-{slot}={value.replace(",", "")}'
                                )
                    self.add_reference(
                        "; ".join(state),
                        dialogue_id=dialog_id,
                        turn_id=f"Turn-{turn_id - 1}",
                    )

        print("\nExtracting the predictions...\n")
        with open(predictions_file, "r") as predictions:
            for line in tqdm(predictions):
                # The predictions csv is composed of the id and the prediction
                fields = line.split(",", 1)
                # Example of id: SNG1751_Turn-26
                dialogue_id = fields[0].split("/")[-1].split("_")[0]
                turn_id = fields[0].split("/")[-1].split("_")[1]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference_manifest",
        type=str,
        help="The path to the reference txt file.",
        default="./data/dev_manifest.txt",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        help="The path where to find the csv file with the models predictions.",
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

    metrics = SpokenWOZDSTEval(slot_categories="./slot_list.json")
    metrics.read_files(
        predictions_file=args.predictions,
        reference_manifest=args.reference_manifest,
        filtering=not args.no_filtering,
    )

    metrics.prepare_samples(evaluate_ci=args.evaluate_ci)

    print(metrics.summary())
