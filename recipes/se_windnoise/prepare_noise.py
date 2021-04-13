import os
import speechbrain as sb
import torchaudio

def glob_all(folder: str, filt: str) -> list:
    """Recursive glob"""
    import os
    import fnmatch
    matches = []
    for root, dirnames, filenames in os.walk(folder, followlinks=True):
        for filename in fnmatch.filter(filenames, filt):
            matches.append(os.path.join(root, filename))
    return matches


def find_wavs(folder: str) -> list:
    """Finds all wavs in folder"""
    wavs = glob_all(folder, '*.wav')

    return wavs

def prepare_csv(filelist, csv_file, max_length=None):
    """Iterate a set of wavs and write the corresponding csv file.

    Arguments
    ---------
    folder : str
        The folder relative to which the files in the list are listed.
    filelist : str
        The location of a file listing the files to be used.
    csvfile : str
        The location to use for writing the csv file.
    max_length : float
        The maximum length in seconds. Waveforms longer
        than this will be cut into pieces.
    """
    try:
        if sb.utils.distributed.if_main_process():
            with open(csv_file, "w") as w:
                w.write("ID,duration,wav,wav_format,wav_opts\n\n")
                for line in open(filelist):

                    # Read file for duration/channel info
                    # filename = os.path.join(folder, line.split()[-1])
                    filename = line.split()[-1]
                    signal, rate = torchaudio.load(filename)

                    # Ensure only one channel
                    if signal.shape[0] > 1:
                        signal = signal[0].unsqueeze(0)
                        torchaudio.save(filename, signal, rate)

                    ID, ext = os.path.basename(filename).split(".")
                    duration = signal.shape[1] / rate

                    # Handle long waveforms
                    if max_length is not None and duration > max_length:
                        # Delete old file
                        os.remove(filename)
                        for i in range(int(duration / max_length)):
                            start = int(max_length * i * rate)
                            stop = int(
                                min(max_length * (i + 1), duration) * rate
                            )
                            new_filename = (
                                filename[: -len(f".{ext}")] + f"_{i}.{ext}"
                            )
                            torchaudio.save(
                                new_filename, signal[:, start:stop], rate
                            )
                            csv_row = (
                                f"{ID}_{i}",
                                str((stop - start) / rate),
                                new_filename,
                                ext,
                                "\n",
                            )
                            w.write(",".join(csv_row))
                    else:
                        w.write(
                            ",".join((ID, str(duration), filename, ext, "\n"))
                        )
    finally:
        sb.utils.distributed.ddp_barrier()

def preapre_noise(folder: str, noise_list_name='wind_noise_list'):

    all_wavs = find_wavs(folder)
    print("find wavs:{}".format(len(all_wavs)))
    with open(noise_list_name, mode='w') as w:
        # w.writelines(all_wavs + '\n')
        for wav_name in all_wavs:
            w.writelines(wav_name + '\n')

    prepare_csv(noise_list_name,'noise.csv', max_length=3.0)

if __name__ == "__main__":
    import argparse

    # preapre_noise('/home/wangwei/work/corpus/dcunet/multi-dcunet-uss-test-data/sr_16k/s4')   
    preapre_noise('/home/wangwei/work/corpus/windnoise/dataset/244807')      
