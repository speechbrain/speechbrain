#!/usr/bin/env python3

import glob
import json
import logging

logging.basicConfig(level=logging.INFO)

with open("tutorials/notebook-header.md", "r", encoding="utf-8") as header_file:
    header_contents = header_file.read()


def find_first_cell_with_tag(cell_list, tag_to_find):
    for cell in cell_list:
        tags = cell.get("metadata", {}).get("tags", {})
        if tag_to_find in tags:
            return cell

    return None


def update_header(header_cell, path):
    header_cell.update(
        {
            "cell_type": "markdown",
            "metadata": {"id": "sb_auto_header", "tags": ["sb_auto_header"]},
            "source": header_contents.replace(
                "{tutorialpath}", path
            ).splitlines(True),
        }
    )


def update_notebook(fname):
    logging.info(f"Updating {fname}")

    tutorial_path = fname.replace("./", "")

    with open(fname) as f:
        nb = json.load(f)

        cells = nb["cells"]
        header_cell = find_first_cell_with_tag(cells, "sb_auto_header")
        if header_cell is None:
            logging.info("Header not found; creating")
            cells.insert(0, {})
            header_cell = cells[0]

        update_header(header_cell, tutorial_path)

    with open(fname, "w") as wf:
        json.dump(nb, wf, indent=1, ensure_ascii=False)
        print(file=wf)  # print final newline that jupyter adds apparently


if __name__ == "__main__":
    for fname in glob.glob("./tutorials/**/*.ipynb", recursive=True):
        update_notebook(fname)
