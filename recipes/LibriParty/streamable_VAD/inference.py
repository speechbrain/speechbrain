#!/usr/bin/env python
"""
Performs realtime Voice Activity Detection (VAD) on a microphone input stream.
At the end of the stream, is computes the offline values and plots the results for comparison.

Authors:
    Francesco Paissan, 2022, 2023
"""
from speechbrain.pretrained import VAD
from torchaudio.io import StreamReader
import matplotlib.pyplot as plt
from drawnow import drawnow
from time import time
import torch
import sys


eps = 1e-8


class RingBuffer:
    """Handles a ring buffer for realtime inference. """

    def __init__(self, size):
        self.queue = []
        self.size = size

    def append(self, x):
        if len(self.queue) < self.size:
            self.queue.append(x)
        else:
            self.queue.pop(0)
            self.queue.append(x)

    def __call__(self):
        return torch.cat(self.queue, axis=1)


if __name__ == "__main__":
    # Load the VAD model
    vad_interface = VAD.from_hparams(
        source="speechbrain/stream-vad-crdnn-libriparty",
        savedir="pretrained_models/stream-vad-crdnn-libriparty",
    )

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
        frames_per_chunk=80,
        buffer_chunk_size=200,
        sample_rate=vad_interface.sample_rate,
    )

    wav_buffer = RingBuffer(5)

    receptive_field = 5
    features_buffer = RingBuffer(
        receptive_field
    )  # used to store features that go into the CNN input

    # Plotting vars
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
    prob_avg = None  # moving average
    avg_alpha = 0.97
    while retry:
        try:
            last_h = None  # RNN hidden representation
            for s in stream.stream():
                if (time() - start_time) > 20:  # every 20s, reset h
                    print("Hidden state reset.")
                    start_time = time()
                    last_h = None

                retry = False
                # every chunk is 5 ms
                wav_buffer.append(s[0].permute(1, 0))

                raw_waveform += list(s[0].squeeze().numpy())
                # print(np.mean(raw_waveform), np.std(raw_waveform))

                if len(wav_buffer.queue) < 5:
                    continue

                if i % 2:
                    # appends buffered 200ms windows with a 10ms overlap
                    streamed_wav.append(torch.cat(wav_buffer.queue, axis=1))

                    features = vad_interface.mods.compute_features(wav_buffer())

                    features = features[0, 1, :].reshape(
                        1, 1, -1
                    )  # take central frame -- due to padding

                    features = vad_interface.mods.mean_var_norm(
                        features, torch.ones([1])
                    )

                    # appends features computed on 200ms windows with 10ms overlap -- central frame
                    streamed_features.append(features)

                    # features computed on 200ms windows with 10ms overlap -- central frame
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
