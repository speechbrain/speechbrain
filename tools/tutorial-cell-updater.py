#!/usr/bin/env python3

import nbformat
import glob
import logging
import json
logging.basicConfig(level=logging.INFO)

def update_notebook(fname):
    logging.info(f"Updating {fname}")

    with open(fname) as f:
        nb = json.load(f)
        
        cells = nb["cells"]
        
        for cell in cells:
            print(list(cell.keys()))

    with open(fname, "w") as wf:
        json.dump(nb, wf, indent=1, ensure_ascii=False)
        print(file=wf) # print final newline that jupyter adds apparently

if __name__ == "__main__":
    for fname in glob.glob("./tutorials/**/*.ipynb", recursive=True):
        update_notebook(fname)
        break
