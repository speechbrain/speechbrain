"""Libraries for automatic finding URLs in the files and checking if they are
reachable.

Authors
 * Mirco Ravanelli 2022
"""

import os
import re
import subprocess
import time
import warnings

import requests

from speechbrain.utils.parallel import parallel_map

DEFAULT_URL_FILE_MATCH_REGEX = r"\.(py|ipynb|md|txt|yaml|yml)$"

# let's ignore lines containing:
# - `ignore-url-check` (just shove it in a comment on the same line)
# - references to `.git` repos (they can be only valid for Git)
# - links to anything localhost (duh)
# - links to arxiv (let's be nice on their mirrors)
# - links to kaggle (links seem to 404 from `requests`... might be intentional. let's not bother them if they don't wanna be bothered)
# - links to the web archive (let's be nice on their mirrors + they shouldn't go down)
DEFAULT_URL_LINE_EXCLUDE_REGEX = r"(ignore-url-check|https?://github\.com/speechbrain/speechbrain\.git|https?://localhost|https?://arxiv\.org/|https?://www\.kaggle\.com|https?://web\.archive\.org)"


def find_urls_in_file(path, line_exclude_regex):
    """This function searches for the URLs in the specified file.

    Arguments
    ---------
    path: path
        Path of the file where to search for URLs.
    line_exclude_regex: Optional[str]
        If a line containing an URL has any match with this regular-expression,
        then the URL is ignored and won't be checked for this line.

    Returns
    -------
    urls: list
        a list of all the URLs found in the specified path.
    """
    # Check if files exist
    if not os.path.exists(path):
        print("File %s not found!" % (path))
        return False

    # Read the file
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()

    lines = text.split("\n")
    if line_exclude_regex is not None:
        lines = [
            line for line in lines if not re.search(line_exclude_regex, line)
        ]
    text = "\n".join(lines)

    # Set up Regex for URL detection
    url_regex = re.compile(
        r"((https?):((//)|(\\\\))+([\w\d:#@%/;$()~_?\+-=\\\.&](#!)?)*)",
        re.DOTALL,
    )
    urls = re.findall(url_regex, text)

    return list(set(urls))


def get_all_urls(file_lst, avoid_urls, line_exclude_regex):
    """This function searches for all the URLs in the specified file list

    Arguments
    ---------
    file_lst: list
        List of the files where to search for URLs.
    avoid_urls: list
        List of URLs to avoid.
    line_exclude_regex: str
        If a line containing an URL has any match with this regular-expression,
        then the URL is ignored and won't be checked for this line.

    Returns
    -------
    urls: dict
        A dictionary where the keys are the detected URLs and the values
        are the files where the URLs are found.
    """
    all_urls = {}

    for path in file_lst:
        if ".gz" in path:
            continue

        urls = find_urls_in_file(path, line_exclude_regex)

        for url in urls:
            # Clean up urls
            url = url[0].split(")")[0]

            # common in jupyter notebook
            if url.endswith("\\n"):
                url = url[:-2]

            if (
                url[-1] == "."
                or url[-1] == ","
                or url[-1] == " "
                or url[-1] == "/"
                or url[-1] == "\\"
            ):
                url = url[:-1]

            if url in avoid_urls:
                continue

            if url not in all_urls:
                all_urls[url] = []
            all_urls[url].append(path)
    return all_urls


def check_url(url, delay=0.5):
    """Checks if an URL is broken. This does NOT perform SSL verification.
    The MITM risk is basically nil here as we don't need to trust the contents,
    and it works around some broken requests for certain websites and
    configurations.

    Arguments
    ---------
    url: string
        URL to check
    delay: float
        Time to wait after a request

    Returns
    -------
    Optional[Any]
        Error that might have occurred. If `None`, then the URL was fetched
        correctly. Either a status code or an exception.
    """
    try:
        with warnings.catch_warnings(action="ignore"):
            response = requests.head(url, verify=False)
        time.sleep(delay)
        if response.status_code == 404 or response.status_code >= 500:
            return response.status_code
        else:
            return None
    except requests.ConnectionError as e:
        return e


def check_links(
    folder=".",
    file_match_regex=DEFAULT_URL_FILE_MATCH_REGEX,
    line_exclude_regex=DEFAULT_URL_LINE_EXCLUDE_REGEX,
    avoid_urls=["http:/", "http://", "https:/", "https://"],
):
    """This test checks if files indexed by git in the given folder contain any
    URL which, when fetched using `requests.head`, returns a 404 error or an
    error code `>= 500`.

    Arguments
    ---------
    folder: path
        The top Folder for searching for the files. This string should be
        trusted as it is not escaped for the shell.
    file_match_regex: Optional[str]
        If a file path has any match with this regular expression, then it is a
        candidate for URL checking.
    line_exclude_regex: Optional[str]
        If a line containing an URL has any match with this regular-expression,
        then the URL is ignored and won't be checked for this line.
    avoid_urls: list
        Exclude URLs that strictly match any of the values in the list.

    Returns
    -------
    check_test: bool
        Whether or not the test is passed.
    """
    check_test = True
    # Find all the files that potentially contain urls
    file_lst = (
        subprocess.check_output(f"git ls-files {folder}", shell=True)
        .decode("utf-8")
        .split("\n")
    )
    print(f"Unfiltered file count: {len(file_lst)}")

    if file_match_regex is not None:
        file_lst = [
            path for path in file_lst if re.search(file_match_regex, path)
        ]
        print(f"Filtered file count: {len(file_lst)}")

    # Get urls for the list of files - unique list
    all_urls = get_all_urls(file_lst, avoid_urls, line_exclude_regex)

    print(f"Found {len(all_urls)} URLs to check")

    for url in all_urls.keys():
        print(url)

    # Check all the urls
    for url, err in zip(
        all_urls.keys(),
        parallel_map(
            check_url, list(all_urls.keys()), chunk_size=1, process_count=8
        ),
    ):

        if err is not None:
            print(f"WARNING: {url} is DOWN! got {err}")
            for path in all_urls[url]:
                print(f"\t link appears in {path}")

        check_test &= err is not None
        time.sleep(0.1)
    return check_test
