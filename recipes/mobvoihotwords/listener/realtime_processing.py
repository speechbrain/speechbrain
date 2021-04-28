import pyaudio
import time
import threading
import wave
import numpy as np

import time
import pyaudio
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from threading import Event

class realtime_processing(object):
    def __init__(self, callback=None, angle=0,chunk=960, channels=6, rate=16000,Recording=False):
        self.CHUNK = chunk
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = channels
        self.RATE = rate
        self._running = True
        self._frames = []
        self.input_device_index = 0
        self.method = 0
        self.callback = callback
        self.angle = angle
        self.isRecording = Recording
        self.playback = False

        self.callback = callable

        self.buffer_time = 10  # ms
        self.buffer_len = int(self.buffer_time*16000/1000)

        self.audio_data = np.zeros((self.buffer_len, 1))

        self.fig, self.ax = plt.subplots()
        self.xdata, self.ydata = [], [] 
        self.plotdata = np.zeros((self.buffer_len, 1))
        self.lines, = self.ax.plot(self.plotdata)

    def audioDevice(self):
        pass

    def start(self):
        if self.isRecording:
            print('Recording...\n')
        threading._start_new_thread(self.__recording, ())

        ani = FuncAnimation(self.fig, self.update_plot, init_func=self.init, blit=True, interval=60)
        #
        plt.show()
        Event().wait()  # Wait forever

    def init(self):
        self.ax.set_xlim(0, len(self.plotdata))
        self.ax.set_ylim(-0.1, 1.0)
        return self.lines,  # 返回曲线

    def update_plot(self, n):
        self.audio_data[:-self.buffer_len] = self.audio_data[self.buffer_len:]
        # self.audio_data[-1] = self.runner.prob
        self.lines.set_ydata(self.audio_data[:, 0])
        return self.lines,

    def process(self, data: np.array):
        self.audio_data[:self.buffer_len - self.CHUNK, 0] = self.audio_data[self.CHUNK:, 0]
        self.audio_data[-self.CHUNK:, 0] = data

    def __recording(self):
        self._running = True
        self._frames = []
        p = pyaudio.PyAudio()
        if self.playback:
            streamOut = p.open(format=self.FORMAT,
                               channels=1,
                                rate=self.RATE,
                                input=False,
                                output=True,
                                # output_device_index=4,
                                frames_per_buffer=self.CHUNK)

        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=self.RATE,
                        input=True,
                        output=False,
                        # output_device_index=4,
                        frames_per_buffer=self.CHUNK)
        while (self._running):
            data = stream.read(self.CHUNK)

            if self.CHANNELS == 6:

                samps = np.fromstring(data, dtype=np.int16)

                # extract channel 0 data from 6 channels, if you want to extract channel 1, please change to [1::6]
                a = np.fromstring(samps, dtype=np.int16)[1::self.CHANNELS] # 1-channel int16
                data = a.tostring()                                        # 1-channel string for playback
                data_float = a.astype(np.float32, order='C') / 32768.0

                self.process(data_float)

                if self.isRecording:
                    MultiChannelPCM = (MultiChannelData * 32768).astype('<i2').tostring()
                    self._frames.append(MultiChannelPCM)

            if self.playback:
                streamOut.write(data, self.CHUNK)  # play back audio stream

        stream.stop_stream()
        stream.close()
        p.terminate()

    def stop(self):
        self._running = False

    def changeAlgorithm(self,index):
        self.method = index

    def save(self, filename):

        p = pyaudio.PyAudio()
        if not filename.endswith(".wav"):
            filename = filename + ".wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(6)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(self._frames))
        wf.close()
        print("Saved")


if __name__ == "__main__":

    rec = realtime_processing()
    print("Start processing...\n")
    rec.start()

    # while True:
    #     a = int(input('"select algorithm: \n0.src  \n1.delaysum  \n2.MVDR  \n3.TFGSC  \n'))
    #     rec.changeAlgorithm(a)
    #     # time.sleep(0.1)

