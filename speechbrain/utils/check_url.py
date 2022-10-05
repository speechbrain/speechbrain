"""Libraries for automatic finding URLs in the files and checking if they are
reachable.

Authors
 * Mirco Ravanelli 2022
"""
import os
import re
import time
import requests
from tqdm.contrib import tqdm
from speechbrain.utils.data_utils import get_all_files


def get_url(path):
    """This function searches for the URLs in the specified file.

    Arguments
    ---------
    path: path
        Path of the file where to search for URLs.

    Returns
    -------
    urls: list
        a list of all the URLs found in the specified path.
    """
    # Check if files exist
    if not (os.path.exists(path)):
        print("File %s not found!" % (path))
        return False

    # Read the file
    with open(path, "r") as file:
        text = file.read()

    # Set up Regex for URL detection
    url_regex = re.compile(
        r"((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)",
        re.DOTALL,
    )
    urls = re.findall(url_regex, text)

    return list(set(urls))


def get_all_urls(file_lst, avoid_urls):
    """This function searches for all the URLs in the specified file list

    Arguments
    ---------
    file_lst: list
        List of the files where to search for URLs.
    avoid_urls: list
        List of URLs to avoid.

    Returns
    -------
    urls: dict
        A dictionary where the keys are the detected URLs and the values
    are the files where the URLs are found.
    """
    all_urls = {}

    for path in file_lst:
        urls = get_url(path)

        for url in urls:

            # Clean up urls
            url = url[0].split(")")[0]
            if (
                url[-1] == "."
                or url[-1] == ","
                or url[-1] == " "
                or url[-1] == "/"
            ):
                url = url[:-1]

            if url in avoid_urls:
                continue

            if url not in all_urls:
                all_urls[url] = []
            all_urls[url].append(path)
    return all_urls


def check_url(url):
    """Cheks if an URL is broken

    Arguments
    ---------
    url: string
        URL to check

    Returns
    -------
    Bool
        False if the URL is broken, True otherwise.
    """
    try:
        response = requests.head(url)
        if response.status_code == 404 or response.status_code > 499:
            return False
        else:
            return True
    except requests.ConnectionError:
        return False


def check_links(
    folder=".",
    match_or=[".py", ".md", ".txt"],
    exclude_or=[".pyc"],
    avoid_files=[""],
    avoid_urls=["http:/", "http://", "https:/", "https://"],
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
    # Find all the files that potentially contain urls
    file_lst = get_all_files(folder, match_or=match_or, exclude_or=exclude_or)

    # Get urls for the list of files - unique list
    all_urls = get_all_urls(file_lst, avoid_urls)

    # Check all the urls
    with tqdm(all_urls) as all_urls_progressbar:
        for url in all_urls_progressbar:
            time.sleep(1)
            if not (check_url(url)):
                check_test = False
                print("WARNING: %s is DOWN!" % (url))
                for path in all_urls[url]:
                    print("\t link detected in %s" % (path))
    return check_test
