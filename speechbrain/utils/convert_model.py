import torch
import argparse
import speechbrain as sb
from collections import OrderedDict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", required=True)
    parser.add_argument("--hparams_key", required=True)
    parser.add_argument("--old_ckpt", required=True)
    parser.add_argument("--new_ckpt", required=True)
    args = parser.parse_args()

    with open(args.hparams) as f:
        hparams = sb.load_extended_yaml(f, overrides={"data_folder": "asdf"})

    ckpt = torch.load(args.old_ckpt)
    assert len(hparams[args.hparams_key].state_dict()) == len(ckpt)

    new_state_dict = OrderedDict()
    for old_key, new_key in zip(ckpt, hparams[args.hparams_key].state_dict()):
        new_state_dict[new_key] = ckpt[old_key]

    torch.save(new_state_dict, args.new_ckpt)
