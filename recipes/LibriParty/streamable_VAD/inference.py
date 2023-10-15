#!/usr/bin/env python
"""
This library enables real-time Voice Activity Detection on microphone input streams.
It also computes offline values and provides visualization for result comparison at the end of the stream.

For instructions on how to run the online inference script, please refer to the README.md.

**Note:** Currently, the PyTorch streamreader and our inference script are compatible with Apple devices only.

Authors:
    Francesco Paissan (2022, 2023)
"""

from speechbrain.pretrained import VAD
from torchaudio.io import StreamReader
import matplotlib.pyplot as plt
from drawnow import drawnow
from time import time
import torch
import sys


# Inference Parameters
frames_per_chunk = 80
buffer_chunk_size = 200
receptive_field = 5
avg_alpha = 0.97  # We smooth the model probabilities with a moving average
reset_time = 20  # We need to reset the model periodically.
# The RNN-based model was trained on sentences of approximately 20 seconds in length.
# Without periodic resets, it tends to perform poorly when processing sequences
# significantly longer than this training limit


class RingBuffer:
    """Handles a ring buffer for real-time inference.

    Arguments
    ---------
        size (int): The size of the ring buffer.
    """

    def __init__(self, size):
        self.queue = []
        self.size = size

    def append(self, x):
        """Appends an element to the ring buffer, removing the oldest element
        if the size is exceeded.

        Arguments:
        ---------
            x: Element to append to the ring buffer.
        """
        if len(self.queue) < self.size:
            self.queue.append(x)
        else:
            self.queue.pop(0)
            self.queue.append(x)

    def __call__(self):
        """Concatenates the elements in the ring buffer along the specified axis.

        Returns:
            torch.Tensor: Concatenated elements in the ring buffer.
        """
        return torch.cat(self.queue, axis=1)


if __name__ == "__main__":

    # Load the VAD model
    vad_interface = VAD.from_hparams(
        source="speechbrain/stream-vad-crdnn-libriparty",
        savedir="pretrained_models/stream-vad-crdnn-libriparty",
    )

    # Setting the VAD in eval mode for inference use.
    vad_interface.eval()

    # get microphone ID
    if len(sys.argv) > 1:
        mic_id = eval(sys.argv[1])
    else:
        mic_id = 0
    print("Using microphone with ID %d." % mic_id)

    # Process the audio stream
    stream = StreamReader(
        src=":%d" % mic_id,  # here you should select the correct input device
        format="avfoundation",
    )

    stream.add_basic_audio_stream(
        frames_per_chunk=frames_per_chunk,
        buffer_chunk_size=buffer_chunk_size,
        sample_rate=vad_interface.sample_rate,
    )

    wav_buffer = RingBuffer(receptive_field)

    # Store features for the CNN input. The buffer length must match the CNN's receptive field.
    features_buffer = RingBuffer(receptive_field)

    # Plotting variables
    raw_waveform = []
    streamed_wav = []
    streamed_features = []
    streamed_rf = []
    streamed_cnn = []
    streamed_rnn = []
    streamed_mlp = []
    probs = []
    i = 1

    def plot_waveform():
        """
        Plot the raw waveform and speech probability in a real-time manner.
        """
        plt.subplot(211)
        plt.ylabel("Raw waveform")
        plt.plot(raw_waveform, color="green")

        plt.subplot(212)
        plt.ylabel("Speech probability")
        plt.plot(probs, color="blue")

    plt.ion()
    fig, ax = plt.subplots(2, 1)

    retry = True
    start_time = time()
    prob_avg = None  # Initializing moving average variable.
    while retry:
        try:
            last_h = None  # RNN hidden representation
            for s in stream.stream():
                if (time() - start_time) > reset_time:  # every 20s, reset h
                    print("Hidden state reset.")
                    start_time = time()
                    last_h = None

                retry = False
                # Every chunk is 5 ms
                wav_buffer.append(s[0].permute(1, 0))

                raw_waveform += list(s[0].squeeze().numpy())

                if len(wav_buffer.queue) < 5:
                    continue

                if i % 2:
                    # Appends buffered 200ms windows with a 10ms overlap
                    streamed_wav.append(torch.cat(wav_buffer.queue, axis=1))

                    features = vad_interface.mods.compute_features(wav_buffer())

                    features = features[0, 1, :].reshape(
                        1, 1, -1
                    )  # Take central frame -- due to padding

                    features = vad_interface.mods.mean_var_norm(
                        features, torch.ones([1])
                    )

                    # Appends features computed on 200ms windows with 10ms overlap -- central frame
                    streamed_features.append(features)

                    # Features computed on 200ms windows with 10ms overlap -- central frame
                    features_buffer.append(features)

                    if len(features_buffer.queue) < receptive_field:
                        continue

                    streamed_rf.append(features_buffer())

                    # CNN encoder
                    outputs = vad_interface.mods["model"][0](features_buffer())

                    # take only the one of the current frame
                    outputs_cnn_stream = outputs[0, -1, :].reshape(1, 1, -1)

                    streamed_cnn.append(outputs_cnn_stream)

                    outputs_rnn_stream, h = vad_interface.mods["model"][1](
                        outputs_cnn_stream, hx=last_h
                    )
                    last_h = h  # update hidden representation

                    streamed_rnn.append(outputs_rnn_stream)

                    outputs = vad_interface.mods["model"][2](outputs_rnn_stream)

                    streamed_mlp.append(outputs)

                    out_prob = torch.sigmoid(outputs).squeeze()

                    if prob_avg is None:
                        prob_avg = out_prob.item()
                    else:
                        prob_avg = out_prob.item() * avg_alpha + prob_avg * (
                            1 - avg_alpha
                        )

                    probs.append(prob_avg)

                    drawnow(plot_waveform)

                i += 1

        except RuntimeError:
            retry = True
        except KeyboardInterrupt:
            retry = False
            drawnow(plot_waveform)
            plt.savefig("streaming.png")

            raw_waveform = torch.Tensor(raw_waveform).unsqueeze(0)
            temp = vad_interface.mods["compute_features"](raw_waveform)
            temp = vad_interface.mods["mean_var_norm"](
                temp, torch.ones([1])
            )  # removed to correct streaming problems with first buffer
            outputs = vad_interface.mods["model"][0](temp)
            outputs_cnn_data = outputs.reshape(
                outputs.shape[0],
                outputs.shape[1],
                outputs.shape[2] * outputs.shape[3],
            )

            outputs_rnn, h = vad_interface.mods["model"][1](outputs_cnn_data)
            outputs_rnn_data = outputs_rnn

            outputs_data = vad_interface.mods["model"][2](
                outputs_rnn_data
            ).squeeze()

            plt.cla()
            fig = plt.subplots(2, 1)

            plt.subplot(211)
            plt.plot(raw_waveform[0], color="red")

            plt.subplot(212)
            plt.plot(torch.sigmoid(outputs_data), color="orange")

            plt.savefig("offline_processing.png")

            plt.clf()
            plt.plot(torch.sigmoid(outputs_data), color="orange")
            plt.plot(probs, color="blue")
            plt.savefig("probs-overlap.png")
