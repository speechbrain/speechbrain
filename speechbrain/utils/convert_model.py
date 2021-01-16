"""This temporary script converts old checkpoints  (i.e., before merging the
Sequential Dict PR - Nov,7 ) into the new format.
You can call this script in this way:

python convert_model.py --hparams current_hyparams.yaml \
--old_ckpt /miniscratch/ravanelm/LM/1234/save/CKPT+2020-08-29+02-30-20+00/model.ckpt \
--new_ckpt /miniscratch/ravanelm/LM_converted/model.ckpt \
--hparams_key model

Authors
 * Peter Plantinga 2020
"""

import torch
import argparse
from collections import OrderedDict
from hyperpyyaml import load_hyperpyyaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--hparams_key", required=True)
    parser.add_argument("--old_ckpt", required=True)
    parser.add_argument("--new_ckpt", required=True)
    args = parser.parse_args()

    with open(args.hparams) as f:
        hparams = load_hyperpyyaml(f, overrides={"data_folder": "asdf"})

    ckpt = torch.load(args.old_ckpt)
    assert len(hparams[args.hparams_key].state_dict()) == len(ckpt)

    new_state_dict = OrderedDict()
    for old_key, new_key in zip(ckpt, hparams[args.hparams_key].state_dict()):
        new_state_dict[new_key] = ckpt[old_key]

    torch.save(new_state_dict, args.new_ckpt)
