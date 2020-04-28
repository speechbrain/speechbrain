"""
 -----------------------------------------------------------------------------
 visualize_pkl.py (author: Mirco Ravanelli)

 Description: This script visualizes a pkl file

 Input:       pkl_lst (list, mandatory):
                 it is a list containing the pkl files  to visualize
                 (passed through command line)

 Output:      None

 Example:     python tools/visualize_pkl.py \
              exp/compute_spectrogram/save/example1.pkl \
              exp/compute_spectrogram/save/example2.pkl \
 -----------------------------------------------------------------------------
"""
import sys
import pickle

if __name__ == "__main__":

    # Try to import matplotlib.pyplot
    try:

        import matplotlib.pyplot as plt

    except Exception:
        err_msg = "cannot import matplotlib. Make sure it is installed."
        raise

    for pkl_file in sys.argv[1:]:

        # Loading pkl tensor
        try:
            with open(pkl_file, "rb") as f:
                tensor = pickle.load(f)
        except Exception:
            err_msg = "Cannot read file %s" % (pkl_file)

        # Flipping the tensor
        tensor = tensor.flip([-2])

        # Visualizing tensor
        plt.imshow(tensor, cmap="hot")
        plt.figure(0)
    plt.show()
