import matplotlib.pyplot as plt

SAMPLE_RATE = 16000
def plot_fbank(writer, spec,i, label, epoch, title=None, ylabel="freq_bin", aspect="auto", xmax=None, filename = None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow((spec.cpu()), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    writer.add_figure(f"Image/{i}_{label}", fig, epoch)
