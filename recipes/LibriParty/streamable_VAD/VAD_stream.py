""" 
Performs real-time Voice Activity Detection (VAD) on a microphone input stream.
At the end of the stream, is computes the offline values and plots the results for comparison.

Authors: 
    Francesco Paissan, 2022
"""
from speechbrain.pretrained import VAD
from torchaudio.io import StreamReader
import matplotlib.pyplot as plt
import torch
from drawnow import drawnow

class RingBuffer:
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
        source=".",
        savedir="VAD_CRDNN"
    )

    vad_interface.eval()

    # Process the audio stream
    stream = StreamReader(
        src=":0",
        format="avfoundation",   
    )
    
    stream.add_basic_audio_stream(
        frames_per_chunk=80,
        buffer_chunk_size=200,
        sample_rate=vad_interface.sample_rate
    )

    wav_buffer = RingBuffer(5)

    receptive_field = 5
    features_buffer = RingBuffer(receptive_field) # used to store features that go into the CNN input

    for _ in range(receptive_field - 1): # so I directly store something in my buffer...
        features_buffer.append(torch.zeros(1, 1, 40))

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
    
    try:
        last_h = None # RNN hidden representation
        for s in stream.stream():
            # every chunk is 5 ms
            wav_buffer.append(s[0].permute(1, 0))

            raw_waveform += list(s[0].squeeze().numpy())

            if len(wav_buffer.queue) < 5:
                continue
            
            if (i % 2):
                # appends buffered 200ms windows with a 10ms overlap
                streamed_wav.append(
                    torch.cat(wav_buffer.queue, axis=1)
                )
                
                features = vad_interface.mods.compute_features(
                    wav_buffer()
                )
                
                features = features[0, 1, :].reshape(1, 1, -1) # take central frame -- due to padding

                # print(
                #     vad_interface.mods.mean_var_norm.glob_mean.data,
                #     vad_interface.mods.mean_var_norm.glob_std.data
                # )
                # input()

                features = vad_interface.mods.mean_var_norm(
                    features, torch.ones([1])
                )

                # print(features)
                # input()
                

                # appends features computed on 200ms windows with 10ms overlap -- central frame
                streamed_features.append(
                    features
                )
                
                # features computed on 200ms windows with 10ms overlap -- central frame
                features_buffer.append(features)
                
                if len(features_buffer.queue) < receptive_field:
                    continue
                        
                streamed_rf.append(
                    features_buffer()
                )
                
                # print(features_buffer().shape)
                # break
                
                # CNN encoder
                outputs = vad_interface.mods["model"][0](features_buffer())
                
                # print(outputs.shape)
                # break
                
                # take only the one of the current frame
                outputs_cnn_stream = outputs[0, -1, :].reshape(1, 1, -1)
                
                # print(outputs_cnn_stream.shape)
                # break
                
                streamed_cnn.append(
                    outputs_cnn_stream
                )
                
                outputs_rnn_stream, h = vad_interface.mods["model"][1](outputs_cnn_stream, hx=last_h)
                last_h = h # update hidden representation
                
                streamed_rnn.append(
                    outputs_rnn_stream
                )
                
                outputs = vad_interface.mods["model"][2](outputs_rnn_stream)
                
                streamed_mlp.append(
                    outputs
                )
                
                probs.append(
                    torch.sigmoid(outputs).squeeze()
                )

                # print(probs)
                drawnow(plot_waveform)

                
            i += 1
    
    except KeyboardInterrupt:
        fig = plt.subplots(2, 1)

        raw_waveform = torch.Tensor(raw_waveform).unsqueeze(0)
        
        temp = vad_interface.mods["compute_features"](raw_waveform)
        temp = vad_interface.mods["mean_var_norm"](temp, torch.ones([1])) # removed to correct streaming problems with first buffer

        outputs = vad_interface.mods["model"][0](temp)
        outputs_cnn_data = outputs.reshape(
            outputs.shape[0],
            outputs.shape[1],
            outputs.shape[2] * outputs.shape[3],
        )

        outputs_rnn, h = vad_interface.mods["model"][1](outputs_cnn_data)

        outputs_rnn_data = outputs_rnn
        outputs_data = vad_interface.mods["model"][2](outputs_rnn_data).squeeze()

        plt.subplot(211)
        plt.plot(raw_waveform[0], color="red")

        plt.subplot(212)
        plt.plot(torch.sigmoid(outputs_data), color="orange")

        plt.savefig("offline_processing.png")



