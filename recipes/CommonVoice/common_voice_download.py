import os
import requests
from tqdm import tqdm
import tarfile
import argparse


_SPLITS = ["train", "dev", "test"]
def download_common_voice(data_folder,language):
    download_dir = os.path.join(data_folder,language)
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)
    os.mkdir(download_dir)
    archive = os.path.join(download_dir, "tmp.tar.gz")
    url = (
    "https://voice-prod-bundler-ee1969a6ce8178826482b88e843c335139bd3fb4.s3.amazonaws.com"
    f"/cv-corpus-13.0-2023-03-09/cv-corpus-13.0-2023-03-09-{language}.tar.gz"
)
    try:
        with requests.get(url, stream=True) as response:
            total_size = int(response.headers.get("content-length", 0))
            chunk_size = 1024 * 1024
            progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
            with open(archive, "wb") as f:
                for data in response.iter_content(chunk_size):
                    progress_bar.update(len(data))
                    f.write(data)
                progress_bar.close()
        # logger.log(logging.INFO, "Done!")

        # logger.log(logging.INFO, "Extracting data...")
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                name = os.path.basename(member.name)
                if name.endswith(".mp3"):
                    member.name = os.path.join(download_dir, "clips", name)
                    tar.extract(member)
                elif os.path.splitext(name)[0] in _SPLITS:
                    member.name = os.path.join(download_dir, name)
                    tar.extract(member)
        os.remove(archive)

    except Exception:
        # shutil.rmtree(download_dir)
        raise RuntimeError(f"Could not download locale: {language}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='download commonvoice 13.')
    parser.add_argument('data_path', type=str,
                        help='folder to save the downloaded data')
    parser.add_argument('--locales', nargs='+',
                        help='languages to downlaod')

    args = parser.parse_args()
    for language in args.locales:
        download_common_voice(args.data_path,language)

