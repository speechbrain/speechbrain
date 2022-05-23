"""Libraries for automatic finding URLs in the files and checking if they are
reachable.

Authors
 * Mirco Ravanelli 2022
"""
import re
import urllib.request
import os
from speechbrain.utils.data_utils import get_all_files


def check_link(path, avoid_links=["https:/", "http:/"]):
    """Extracts all the URLs from the given file and checks if they are reachable
    or not.

    Arguments
    ---------
    path: path
        Path of a file that might contain URLs.
    avoid_links: list
        List containing all the links to avoid.

    Returns
    -------
    Bool
        This function returns True if all the links are reachable and False
        otherwise.
    """

    # Check if files exist
    if not (os.path.exists(path)):
        print("File %s not found!" % (path))
        return False

    # Read the file
    with open(path, "r") as file:
        text = file.read()

    # Set up Regex for URL detection
    link_regex = re.compile(
        r"((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)",
        re.DOTALL,
    )
    links = re.findall(link_regex, text)

    # Check if URL is reachable
    check_link = True

    for lnk in links:
        # Clean up links
        lnk = lnk[0].split(")")[0]
        if lnk[-1] == "." or lnk[-1] == "," or lnk[-1] == " " or lnk[-1] == "/":
            lnk = lnk[:-1]
        # Check if link is to avoid
        if lnk in avoid_links:
            continue
        # Check if url is reachable
        try:
            status_code = urllib.request.urlopen(lnk).getcode()
        except Exception:
            print("Checking %s..." % (path))
            print("\tWARNING:%s is DOWN!" % (lnk))
            check_link = False
            continue
        if status_code != 200:
            print("Checking %s..." % (path))
            print("\tWARNING:%s is DOWN!" % (lnk))
            check_link = False

    return check_link


def test_links(
    folder=".",
    match_or=[".py", ".md", ".txt"],
    exclude_or=[".pyc"],
    avoid_files=[""],
):
    """This test checks if the files in the specified folders contain broken URLs

    Arguments
    ---------
    folder: path
        The top Folder for searching for the files.
    match_or: list
        Used to specify the extensions of the files to check.
    exclude_or: list
        Used to avoid some file extensions.
    avoid_files: list
        Used to avoid testing some specific file.
    """
    check_test = True
    file_lst = get_all_files(folder, match_or=match_or, exclude_or=exclude_or)
    for path in file_lst:
        if path not in avoid_files:
            if not check_link(path):
                check_test = False
    assert check_test
