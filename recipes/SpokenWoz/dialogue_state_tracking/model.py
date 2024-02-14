#!/usr/bin/env python3
"""
Model for cascade and/or end-to-end spoken Dialogue State Tracking (DST).
Overrides the one from MultiWoz to write prediction according to spokenWoz id format.

Author
    * Lucas Druart 2024
"""
import os

from ...MultiWOZ.dialogue_state_tracking.model import DialogueUnderstanding

class SpokenWozUnderstanding(DialogueUnderstanding):

    def write_predictions(self, hyps, batch):
        """
        Overriding the write_predictions method to match the id format from SpokenWoz.
        """
        # Writing the predictions in a file for future evaluation
        with open(self.hparams.output_file, "a") as pred_file:
            for hyp, element_id in zip(hyps, batch.id):
                pred_file.write(
                    f"{element_id},{self.tokenizer.decode(hyp)}\n"
                )

                # Keeping track of the last predicted state of each dialogue to use it for the next prediction
                if not self.hparams.gold_previous_state:
                    # Id in the form /path/to/dialogue/Turn-N
                    dialog_id = element_id.split("/")[-2]
                    state = self.tokenizer.decode(hyp)
                    with open(
                        os.path.join(
                            self.hparams.output_folder,
                            "last_turns",
                            f"{dialog_id}.txt",
                        ),
                        "w",
                    ) as last_turn:
                        last_turn.write(state + "\n")
