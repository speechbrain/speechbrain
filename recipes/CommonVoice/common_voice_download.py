import os
import requests
from tqdm import tqdm
import tarfile
import argparse
import logging

logger = logging.getLogger(__name__)

_SPLITS = ["train", "dev", "test"]
def download_common_voice(download_dir,language):
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)
    archive = os.path.join(download_dir, f"cv-corpus-13.0-2023-03-09-{language}.tar.gz")
    url = (
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com"
    f"/cv-corpus-14.0-2023-06-23/cv-corpus-14.0-2023-06-23-{language}.tar.gz"
)
    try:
        logger.log(logging.INFO, f"start downloading {language}")

        with requests.get(url, stream=True) as response:
            total_size = int(response.headers.get("content-length", 0))
            chunk_size = 1024 * 1024
            progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
            with open(archive, "wb") as f:
                for data in response.iter_content(chunk_size):
                    progress_bar.update(len(data))
                    f.write(data)
                progress_bar.close()
        logger.log(logging.INFO, f"Complete downloading {language}")

    except Exception:
        raise RuntimeError(f"Could not download locale: {language}")

def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='download commonvoice 13.')
    parser.add_argument('data_path', type=str,
                        help='folder to save the downloaded data')
    parser.add_argument('--locales', nargs='+',
                        help='languages to downlaod')

    args = parser.parse_args()
    for language in args.locales:
        download_common_voice(args.data_path,language)

