import argparse

from tqdm import tqdm

from speechbrain.utils.evaluate_dialogue_state_tracking import DSTEval


class MultiWOZDSTEval(DSTEval):
    def read_files(
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

    metrics = MultiWOZDSTEval(slot_categories="./slot_list.json")
    metrics.read_files(
        predictions_file=args.predictions,
        reference_manifest=args.reference_manifest,
        filtering=not args.no_filtering,
    )

    metrics.prepare_samples(evaluate_ci=args.evaluate_ci)

    print(metrics.summary())
