"""Low-level feature pipeline components

This library gathers functions that process batches for data.
All the classes are of type nn.Module. This gives the
possibility to have end-to-end differentiability and to 
backpropagate the gradient through them.
----------------------------------------------------------------
"""

# Importing libraries
import math
import torch
import logging
from speechbrain.data_io.data_io import recovery, initialize_with
logger = logging.getLogger(__name__)


class STFT(torch.nn.Module):
    """
     -------------------------------------------------------------------------
     data_processing.STFT (author: Mirco Ravanelli)

     Description: This class computes the Short-Term Fourier Transform of an
                  audio signal. It supports multi-channel audio inputs. It is
                  a modification of the STFT of the torch audio toolkit
                  (https://github.com/pytorch/audio).

     Input: - sample_rate (type:int(0,inf),mandatory):
               it is the sample rate of the input audio signal

           - device('cuda','cpu',optional,default:None):
               it is the device where to compute the STFT.
               If None, it uses the device of the input signal

           - win_length (type: float(0,inf), opt, default:25):
               it is the length (in ms) of the sliding window
               used to compute the STFT.

           - hop_length (type: float(0,inf), optional,
             default:10):
               it is the length (in ms) of the hope of the
               sliding window used to compute the STFT.

           - n_fft (type:int(0,inf), optional, default:400):
              it the number of fft point of the STFT. It
             defines the frequency resolution (n_fft should
             be <= than win_len).

           - window_type ('bartlett','blackman','hamming',
                          'hann', optional, default: hamming):
               it is the window function used to compute the
               STFT.

           - normalized_stft (type:bool,optional,
             default:False):
               if normalized is True (default is False), the
               function returns the normalized STFT results,
               i.e., multiplied by win_length^-0.5.

           - center: (type:bool,optional, default:True):
               if center is True (default), input will be
               padded on both sides so that the t-th frame is
               centered at time t×hop_length. Otherwise, the
               t-th frame begins at time t×hop_length.

           - pad_mode: ('constant','reflect','replicate',
                        'circular', optional default:reflect):
               It determines the padding method used on input
               when center is True. 'constant' pads the input
               tensor boundaries with a constant value.
               'reflect' pads the input tensor using the
               reflection of the input boundary. 'replicate'
               pads the input tensor using replication of the
               input boundary. 'circular' pads using  circular
               replication.

           - onesided: (type:bool,optional,default:True)
               if True only return nfft/2 values. Note that
               the other samples are redundant due to the
                Fourier transform conjugate symmetry.

      Example:  import torch
                import soundfile as sf
                from data_processing import STFT

                # reading an audio signal
                signal, fs=sf.read('samples/audio_samples/example1.wav')

                # config dictionary definition
                config={'class_name':'data_processing.STFT',
                       'sample_rate':str(fs)}

                # Initialization of the class
                compute_stft=STFT(config)

                # Executing computations
                stft_out=compute_stft(torch.tensor(signal).unsqueeze(0))
                print(stft_out)
                print(stft_out[0].shape)

     -------------------------------------------------------------------------
     """

    def __init__(
        self,
        sample_rate,
        win_length=25,
        hop_length=10,
        n_fft=400,
        window_type='hamming',
        normalized_stft=False,
        center=True,
        pad_mode='constant',
        onesided=True,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.window_type = window_type
        self.normalized_stft = normalized_stft
        self.center = center
        self.pad_mode = pad_mode
        self.onesided = onesided

        # Convert win_length and hop_length from ms to samples
        self.win_length = int(
            round((self.sample_rate / 1000.0) * self.win_length)
        )
        self.hop_length = int(
            round((self.sample_rate / 1000.0) * self.hop_length)
        )

        # Window creation
        self.window = self.create_window()

    def forward(self, x):
        """
        Input: - x (type: torch.Tensor, mandatory):
                   torch.tensor with the audio samples. The tensor must be
                    in one of the following formats: [batch,samples],
                   [batch,channels,samples]

        Output: - complex spectrogram (type: torch.Tensor)
                   The tensor is formatted in one of the
                   following ways depending on the input shape:
                   [batch,n_fft/2, 2, time_steps],
                   [batch,channels,n_fft/2, 2, time_steps]
        """

        # Managing multi-channel stft:
        or_shape = x.shape

        # Reshaping tensor to (batch*channel,time) if needed
        if len(or_shape) == 3:
            x = x.view(or_shape[0] * or_shape[1], or_shape[2])

        # STFT computation
        stft = torch.stft(
            x,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window.to(x.device),
            self.center,
            self.pad_mode,
            self.normalized_stft,
            self.onesided,
        )

        # Retrieving the original dimensionality (batch,channel,time)
        if len(or_shape) == 3:
            stft = stft.reshape(
                or_shape[0],
                stft.shape[1],
                or_shape[1],
                stft.shape[2],
                stft.shape[3],
            )

        # Batch is the first dim, time steps always the last one
        stft = stft.transpose(-2, -1)

        return stft

    def create_window(self):
        """
         ---------------------------------------------------------------------
         core.data_processing.create_window (author: Mirco Ravanelli)

         Description: This function creates the window used to compute the
                      STFT given the type and the w_lenth specified in the
                      config file.

         Input:        - self (type, loop class, mandatory)

         Output:      - window (type, torch.tensor)

         Example:    from data_processing import STFT

                     # config dictionary definition
                    config={'class_name':'data_processing.STFT',
                           'sample_rate':'16000'}

                    # initialization of the class
                    compute_stft=STFT(config)

                    # window creation
                    window=compute_stft.create_window()
                    print(window)
                    print(window.shape)

         ---------------------------------------------------------------------
         """

        # Selecting the window type
        if self.window_type == "bartlett":
            wind_cmd = torch.bartlett_window

        if self.window_type == "blackman":
            wind_cmd = torch.blackman_window

        if self.window_type == "hamming":
            wind_cmd = torch.hamming_window

        if self.window_type == "hann":
            wind_cmd = torch.hann_window

        # Window creation
        window = wind_cmd(self.win_length)

        return window


class spectrogram(torch.nn.Module):
    """
     -------------------------------------------------------------------------
     data_processing.spectrogram (author: Mirco Ravanelli)

     Description: This class computes spectrogram of an audio
                  signal given its STFT in input. It is a
                  modification of the spectrogram of the torch audio toolkit
                  (https://github.com/pytorch/audio).

     Input: - power_spectrogram (type: float, default:2):
                It is the exponent used for spectrogram
                computation.  By default, we compute the power
                spectrogram (power_spectrogram=2)

     Example:   import torch
                import soundfile as sf
                from data_processing import STFT
                from data_processing import spectrogram

                # reading an audio signal
                signal, fs=sf.read('samples/audio_samples/example1.wav')

                # Initialization of the class
                compute_stft=STFT(sample_rate=fs)

                # Executing computations
                stft_out=compute_stft(torch.tensor(signal).unsqueeze(0))

                # Initialization of the class
                compute_spectr=spectrogram()

                # Executing computations
                spectr_out=compute_spectr(stft_out)
                print(spectr_out)
                print(spectr_out[0].shape)
     -------------------------------------------------------------------------
     """

    def __init__(
        self,
        power_spectrogram=2,
    ):
        super().__init__()

        self.power_spectrogram = power_spectrogram

    def forward(self, stft):
        """
        Input: - stft (type: torch.Tensor, mandatory):
                   The input STFT tensor must be in one of the
                   following formats: [batch, n_fft/2, 2, time_steps],
                   [batch, channels, n_fft/2, 2, time_steps]


        Output: - spectrogram (type: torch.Tensor):
                   The tensor is formatted in one of
                   the following ways depending on the input shape:
                   [batch,n_fft/2, time_steps],
                   [batch,channels,n_fft/2,time_steps]
        """
        # Get power of "complex" tensor (index=-2 are real and complex parts)
        spectrogram = stft.pow(self.power_spectrogram).sum(-2)

        return spectrogram


class FBANKs(torch.nn.Module):
    """
     -------------------------------------------------------------------------
     data_processing.FBANKs (author: Mirco Ravanelli)

     Description: This class computes FBANK features of an audio signal given
                  its spectrogram in input. It is a modification of the FBANKs
                  funct of the torch audio toolkit
                  (https://github.com/pytorch/audio).

     Input: - n_mels (type:int(1,inf),optional,default:40):
               it the number of Mel fiters used to average the
               spectrogram.

           - log_mel (type:bool, optional, default:True):
               if True, it computes the log of the FBANKs

           - filter_shape (triangular,rectangular,gaussian,
                           optional,default:triangular),
               it is the shape of the filters used to compute
               the FBANK filters.

           - f_min (type:float(0,inf),optional, default:0)
               it is the lowest frequency for the Mel filters.

           - f_max (type:float(0,inf),optional, default:8000)
               it is the highest freq for the Mel filters.

           - n_fft (type:int(0,inf), optional, default:400):
              it the number of fft point of the STFT. It
             defines the frequency resolution (n_fft should be
              <= than win_len).

           - sample_rate (type:int(0,inf),mandatory):
               it is the samplerate of the input audio signal.

           - power_spectrogram (type:float,optional,
             default:2):
               It is the exponent used for spectrogram
               computation. By default, we compute the power
                spectrogram (power_spectrogram=2).

           - amin (type: float, optional, default: 1e-10)
               it is the minimum amplitude (used for numerical
               stability).

           - ref_value (type: float, optional, default: 1.0)
               it is the refence value use for the dB scale.

           - top_db (type: float, optional, default: 80)
               it is the top dB value.

           - freeze (type: bool, optional, True)
               if False, it the central frequency and the band
               of each filter are added into nn.parameters
               can be trained. If True, the standard frozen
               features are computed.
           - recovery (type: bool, optional, True)
               this option is used to recover the last filter
               banks parameters saved during training. It is
               activated only if freeze=False.

           - initialize_with (type: str, optional, None)
              this option is a path to a pkl file that
              contains personalized bands and central
              frequnecy for each filter.

     Example:   import torch
                import soundfile as sf
                from data_processing import STFT
                from data_processing import spectrogram
                from data_processing import FBANKs

                # reading an audio signal
                signal, fs=sf.read('samples/audio_samples/example1.wav')

                # Initialization of the STFT class
                compute_stft=STFT(sample_rate=fs)

                # Executing computations
                stft_out=compute_stft(torch.tensor(signal)\
                         .unsqueeze(0).float())

                # Initialization of the spectrogram class
                compute_spectr=spectrogram()

                # Computation of the spectrogram
                spectr_out=compute_spectr(stft_out)

                Initialization of the FBANK class
                compute_fbank=FBANKs(config)

                # Executing computations
                fbank_out=compute_fbank(spectr_out)

                # Executing computations
                fbank_out=compute_fbank(spectr_out)
                print(fbank_out)
                print(fbank_out[0].shape)
     --------------------------------------------.----------------------------
     """

    def __init__(
        self,
        n_mels=40,
        log_mel=True,
        filter_shape='triangular',
        f_min=0,
        f_max=8000,
        n_fft=400,
        sample_rate=16000,
        power_spectrogram=2,
        amin=1e-10,
        ref_value=1.,
        top_db=80.,
        freeze=True,
        recovery=True,
        initialize_path=None,
    ):
        super().__init__()

        self.n_mels = n_mels
        self.log_mel = log_mel
        self.filter_shape = filter_shape
        self.f_min = f_min
        self.f_max = f_max
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.power_spectrogram = power_spectrogram
        self.amin = amin
        self.ref_value = ref_value
        self.top_db = top_db
        self.freeze = freeze
        self.recovery = recovery
        self.initialize_with = initialize_path

        # Additional options
        self.n_stft = self.n_fft // 2 + 1

        # Make sure that the selected f_min < f_max
        if self.f_min >= self.f_max:
            err_msg = "Require f_min: %f < f_max: %f" % (
                self.f_min,
                self.f_max,
            )
            logger.error(err_msg, exc_info=True)

        # Setting the multiplier for log conversion
        if self.power_spectrogram == 2:
            self.multiplier = 10
        else:
            self.multiplier = 20

        self.db_multiplier = math.log10(max(self.amin, self.ref_value))

        # Filter definition
        mel = torch.linspace(
            self.to_mel(self.f_min), self.to_mel(self.f_max), self.n_mels + 2
        )

        # Conversion to hz
        hz = self.to_hz(mel)

        # Computation of the filter bands
        band = hz[1:] - hz[:-1]

        self.band = band[:-1]
        self.f_central = hz[1:-1]

        # Adding the central frequency and the band to the list of nn param
        if not self.freeze:
            self.f_central = nn.Parameter(self.f_central)
            self.band = nn.Parameter(self.band)

        # Frequency axis
        all_freqs = torch.linspace(0, self.sample_rate // 2, self.n_stft)

        # replicating for all the filters
        self.all_freqs_mat = all_freqs.repeat(self.f_central.shape[0], 1)

        def hook(self, first_input):
            self.device_inp = first_input[0].device
            self.band = self.band.to(self.device_inp)
            self.f_central = self.f_central.to(self.device_inp)
            self.all_freqs_mat = self.all_freqs_mat.to(self.device_inp)
            self.hook.remove()

        self.hook = self.register_forward_pre_hook(hook)

        # Managing initialization with an external filter bank parameters
        # (useful for pre-training)
        initialize_with(self)

        if not (self.freeze):
            # Automatic recovery
            recovery(self)

    def forward(self, spectrogram):

        # Getting the current device
        self.device_inp = spectrogram.device

        # Putting all_freq tensor in the current device
        self.all_freqs_mat = self.all_freqs_mat.to(self.device_inp)
        self.band = self.band.to(self.device_inp)
        self.f_central = self.f_central.to(self.device_inp)

        # Computing central frequency and bandwidth of each filter
        f_central_mat = self.f_central.repeat(
            self.all_freqs_mat.shape[1], 1
        ).transpose(0, 1)
        band_mat = self.band.repeat(self.all_freqs_mat.shape[1], 1).transpose(
            0, 1
        )

        # Creation of the multiplication matrix
        fbank_matrix = self.create_fbank_matrix(f_central_mat, band_mat).to(
            spectrogram.device
        )

        # FBANK computation
        fbanks = torch.matmul(
            spectrogram.transpose(1, -1), fbank_matrix
        ).transpose(1, -1)

        # Add logarithm if needed
        if self.log_mel:
            fbanks = self.amplitude_to_DB(fbanks)

        return fbanks

    @staticmethod
    def to_mel(hz):
        """
         ---------------------------------------------------------------------
         core.data_procesing.to_mel

         Description: This function allows a conversion from Hz to Mel

         Input:        - Hz (type: float, mandatory)

         Output:      - Mel (type: float)

         Example:     from data_processing import FBANKs

                      # FBANK config dictionary definition
                      config={'class_name':'data_processing.FBANKs'}

                     # Initialization of the FBANK class
                     compute_fbank=FBANKs(config)

                     # Conversion to mel scale
                     compute_fbank.to_hz(4000)
         ---------------------------------------------------------------------
         """
        return 2595 * math.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        """
         ---------------------------------------------------------------------
         core.data_procesing.to_hz

         Description: This function allows a conversion from Mel scale to Hz

         Input:        - self (type: data_processing class, mandatory)

         Output:      - Hz (type: float)

         Example:     from data_processing import FBANKs

                      # FBANK config dictionary definition
                      config={'class_name':'data_processing.FBANKs'}

                     # Initialization of the FBANK class
                     compute_fbank=FBANKs(config)

                     # Conversion to mel scale
                     compute_fbank.to_mel(23651.39)
         ---------------------------------------------------------------------
         """
        return 700 * (10 ** (mel / 2595) - 1)

    def triangular_filters(self, all_freqs, f_central, band):
        """
         ---------------------------------------------------------------------
         core.data_procesing.triangular_filters

         Description: This function creates triangular filter banks

         Input:        - self (type: data_processing class, mandatory)
                       - all_freqs (type: torch.tensor, mandatory)
                       - f_central (type: torch.tensor, mandatory)
                       - band (type: torch.tensor, mandatory)

         Output:      - fbank_matrix (type, torch.tensor)

         Example:     import torch
                      from data_processing import FBANKs

                      # FBANK config dictionary definition
                      config={'class_name':'data_processing.FBANKs'}

                      # Initialization of the FBANK class
                      compute_fbank=FBANKs(config)

                     # All frequency tensor
                     all_freqs=torch.linspace(0, 8000, 201).repeat(40,1)

                     f_central_mat=compute_fbank.f_central.repeat(
                                   all_freqs.shape[1],1).transpose(0,1)

                     band_mat=compute_fbank.band.repeat(
                              all_freqs.shape[1],1).transpose(0,1)

                     # Compute triangular filters
                     filters=compute_fbank.triangular_filters(
                             all_freqs,f_central_mat,band_mat)

                     print(filters)
                     print(filters.shape)

         ---------------------------------------------------------------------
         """

        # Computing the slops of the filters
        slope = (all_freqs - f_central) / band

        # Left part of the filter
        left_side = slope + 1.0

        # Right part of the filter
        right_side = -slope + 1.0

        zero = torch.zeros(1).to(self.device_inp)

        # Adding zeros for negative values
        fbank_matrix = torch.max(
            zero, torch.min(left_side, right_side)
        ).transpose(0, 1)

        return fbank_matrix

    def rectangular_filters(self, all_freqs, f_central, band):
        """
         ---------------------------------------------------------------------
         core.data_procesing.rectangular_filters

         Description: This function creates rectangular filter banks

         Input:        - self (type: data_processing class, mandatory)
                       - all_freqs (type: torch.tensor, mandatory)
                       - f_central (type: torch.tensor, mandatory)
                       - band (type: torch.tensor, mandatory)

         Output:      - fbank_matrix (type, torch.tensor)

         Example:     import torch
                      from data_processing import FBANKs

                      # FBANK config dictionary definition
                      config={'class_name':'data_processing.FBANKs'}

                      # Initialization of the FBANK class
                      compute_fbank=FBANKs(config)

                     # All frequency tensor
                     all_freqs=torch.linspace(0, 8000, 201).repeat(40,1)

                     f_central_mat=compute_fbank.f_central.repeat(
                                   all_freqs.shape[1],1).transpose(0,1)

                     band_mat=compute_fbank.band.repeat(
                              all_freqs.shape[1],1).transpose(0,1)

                     # Compute rectangular filters
                     filters=compute_fbank.rectangular_filters(
                             all_freqs,f_central_mat,band_mat)

                     print(filters)
                     print(filters.shape)

         ---------------------------------------------------------------------
         """

        # Low cut-off frequency of the filters
        low_hz = f_central - band

        # High cut-off frequency of the filters
        high_hz = f_central + band

        # Left part of the filter
        left_side = right_size = all_freqs.ge(low_hz)

        # Right part of the filter
        right_size = all_freqs.le(high_hz)

        # Computing fbank matrix
        fbank_matrix = (left_side * right_size).float().transpose(0, 1)

        return fbank_matrix

    def gaussian_filters(self, all_freqs, f_central, band, smooth_factor=2):
        """
         ---------------------------------------------------------------------
         core.data_procesing.gaussian_filters

         Description: This function creates gaussian filter banks

         Input:        - self (type: data_processing class, mandatory)
                       - all_freqs (type: torch.tensor, mandatory)
                       - f_central (type: torch.tensor, mandatory)
                       - band (type: torch.tensor, mandatory)

         Output:      - fbank_matrix (type: torch.tensor)

         Example:     import torch
                      from data_processing import FBANKs

                      # FBANK config dictionary definition
                      config={'class_name':'data_processing.FBANKs'}

                      # Initialization of the FBANK class
                      compute_fbank=FBANKs(config)

                     # All frequency tensor
                     all_freqs=torch.linspace(0, 8000, 201).repeat(40,1)

                     f_central_mat=compute_fbank.f_central.repeat(
                                   all_freqs.shape[1],1).transpose(0,1)

                     band_mat=compute_fbank.band.repeat(
                              all_freqs.shape[1],1).transpose(0,1)

                     # Compute gaussian filters
                     filters=compute_fbank.gaussian_filters(
                             all_freqs,f_central_mat,band_mat)

                     print(filters)
                     print(filters.shape)

         ---------------------------------------------------------------------
         """

        # computing the fbank matrix with gaussian shapes
        fbank_matrix = torch.exp(
            -0.5 * ((all_freqs - f_central) / (band / smooth_factor)) ** 2
        ).transpose(0, 1)

        return fbank_matrix

    def create_fbank_matrix(self, f_central_mat, band_mat):
        """
        ---------------------------------------------------------------------
        core.data_procesing.create_fbank_matrix

        Description: This function creates a set filter as specified in the
                      configuration file.

        Input:        - self (type: data_processing class, mandatory)
                       - f_central (type: torch.tensor, mandatory)
                       - band (type: torch.tensor, mandatory)

        Output:      - fbank_matrix (type: torch.tensor)

        Example:     import torch
                      from data_processing import FBANKs

                      # FBANK config dictionary definition
                      config={'class_name':'data_processing.FBANKs'}

                      # Initialization of the FBANK class
                      compute_fbank=FBANKs(config)

                     # getting central frequency and band
                     f_central_mat=compute_fbank.f_central.repeat(
                                   all_freqs.shape[1],1).transpose(0,1)

                     band_mat=compute_fbank.band.repeat(
                              all_freqs.shape[1],1).transpose(0,1)

                     # Compute gaussian filters
                     filters=compute_fbank.create_fbank_matrix(f_central_mat,
                                                               band_mat)

                     print(filters)
                     print(filters.shape)

        ---------------------------------------------------------------------
        """
        # Triangular filters
        if self.filter_shape == "triangular":
            fbank_matrix = self.triangular_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )
        # Rectangular filters
        if self.filter_shape == "rectangular":
            fbank_matrix = self.rectangular_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )
        # Gaussian filters
        if self.filter_shape == "gaussian":
            fbank_matrix = self.gaussian_filters(
                self.all_freqs_mat, f_central_mat, band_mat
            )

        return fbank_matrix

    def amplitude_to_DB(self, x):
        """
         ---------------------------------------------------------------------
         core.data_procesing.amplitude_to_DB

         Description: This function takes in input a set of linear FBANKs and
                      and converts them in log-FBANKs

         Input:        - self (type: execute_computaion class, mandatory)
                       - x (type: )

         Output:      - fbank_matrix (type: torch.tensor)

         Example:     import torch
                      from data_processing import FBANKs

                      # FBANK config dictionary definition
                      config={'class_name':'data_processing.FBANKs'}

                      # Initialization of the FBANK class
                      compute_fbank=FBANKs(config)

                     # getting central frequency and band
                     f_central_mat=compute_fbank.f_central.repeat(
                                   all_freqs.shape[1],1).transpose(0,1)

                     band_mat=compute_fbank.band.repeat(
                              all_freqs.shape[1],1).transpose(0,1)

                     # Compute gaussian filters
                     filters=compute_fbank.create_fbank_matrix(f_central_mat,
                                                               band_mat)

                     print(filters)
                     print(filters.shape)

         ---------------------------------------------------------------------
         """

        # Converting x to the log domain
        x_db = self.multiplier * torch.log10(torch.clamp(x, min=self.amin))
        x_db -= self.multiplier * self.db_multiplier

        # Setting up dB max
        new_x_db_max = torch.tensor(
            float(x_db.max()) - self.top_db, dtype=x_db.dtype, device=x.device,
        )
        # Clipping to dB max
        x_db = torch.max(x_db, new_x_db_max)

        return x_db


class MFCCs(torch.nn.Module):
    """
     -------------------------------------------------------------------------
     data_processing.MFCCs (author: Mirco Ravanelli)

     Description:  This class computes MFCCs of an audio signal given a set of
                   FBANKs in input. It is a modification of the spectrogram of
                    the torch audio toolkit (https://github.com/pytorch/audio)

     Input: - n_mels (type: int(1,inf), default: 40):
               it the number of Mel fiters used to average the spectrogram.

           - dct_norm ('ortho', 'None', default: 'ortho'):
               it is the type of dct transform used to compute MFCCs.

     Example:   import torch
                import soundfile as sf
                from data_processing import STFT
                from data_processing import spectrogram
                from data_processing import FBANKs
                from data_processing import MFCCs

                # reading an audio signal
                signal, fs=sf.read('samples/audio_samples/example1.wav')

                # Initialization of the STFT class
                compute_stft=STFT(sample_rate=fs)

                # Executing computations
                stft_out=compute_stft(torch.tensor(signal)\
                         .unsqueeze(0).float())

                # Initialization of the spectrogram class
                compute_spectr=spectrogram()

                # Computation of the spectrogram
                spectr_out=compute_spectr(stft_out)

                Initialization of the FBANK class
                compute_fbank=FBANKs()

                # Executing computations
                fbank_out=compute_fbank(spectr_out)

                Initialization of the MFCC class
                compute_mfccs=MFCCs()

                # Executing computations
                mfcc_out=compute_mfccs(fbank_out)
                print(mfcc_out)
                print(mfcc_out[0].shape)
     -------------------------------------------------------------------------
     """

    def __init__(
        self,
        n_mfcc=20,
        n_mels=40,
        dct_norm='ortho',
    ):
        super().__init__()

        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.dct_norm = dct_norm

        # Generate matix for DCT transformation
        self.dct_mat = self.create_dct()

        def hook(self, first_input):
            self.device_inp = first_input[0].device
            self.dct_mat = self.dct_mat.to(self.device_inp)
            self.hook.remove()

        self.hook = self.register_forward_pre_hook(hook)

        # Check n_mfcc
        if self.n_mfcc > self.n_mels:

            err_msg = (
                "Cannot select more MFCC coefficients than mel filters "
                "(n_mfcc=%i, n_mels=%i)" % (self.n_mfcc, self.n_mels)
            )

            logger.error(err_msg, exc_info=True)

    def create_dct(self):

        """
         ---------------------------------------------------------------------
         core.data_procesing.create_dct

         Description: This function generate the dct matrix from MFCC
                      computation.

         Input:        - self (type: execute_computaion class, mandatory)

         Output:      - dct (type: torch.tensor)

         Example:   import torch
                    import soundfile as sf
                    from data_processing import STFT
                    from data_processing import spectrogram
                    from data_processing import FBANKs
                    from data_processing import MFCCs

                    # reading an audio signal
                    signal, fs=sf.read('samples/audio_samples/example1.wav')

                    # STFT config dictionary definition
                    config={'class_name':'data_processing.STFT',\
                            'sample_rate':str(fs)}

                    # Initialization of the STFT class
                    compute_stft=STFT(config)

                    # Executing computations
                    stft_out=compute_stft(torch.tensor(signal)\
                             .unsqueeze(0).float())

                    # Initialization of the spectrogram config
                    config={'class_name':'data_processing.spectrogram'}

                    # Initialization of the spectrogram class
                    compute_spectr=spectrogram(config)

                    # Computation of the spectrogram
                    spectr_out=compute_spectr(stft_out)

                    # FBANK config dictionary definition
                    config={'class_name':'data_processing.FBANKs'}

                    Initialization of the FBANK class
                    compute_fbank=FBANKs(config)

                    # Executing computations
                    fbank_out=compute_fbank(spectr_out)

                    # MFCC config dictionary definition
                    config={'class_name':'data_processing.MFCCs'}

                    Initialization of the MFCC class
                    compute_mfccs=MFCCs(config)

                    # Executing computations
                    print(compute_mfccs.create_dct())

     -------------------------------------------------------------------------
     """

        n = torch.arange(float(self.n_mels))
        k = torch.arange(float(self.n_mfcc)).unsqueeze(1)
        dct = torch.cos(math.pi / float(self.n_mels) * (n + 0.5) * k)

        if self.dct_norm is None:
            dct *= 2.0
        else:
            assert self.dct_norm == "ortho"
            dct[0] *= 1.0 / math.sqrt(2.0)
            dct *= math.sqrt(2.0 / float(self.n_mels))

        return dct.t()

    def forward(self, fbanks):
        """
        Input: - (type, list, mandatory):
                   Contains the FBANKs of an audio
                   signal. The input FBANK tensor must be in one of the
                   following ways: [batch, n_mels, time_steps],
                   [batch, channels, n_mels, time_steps]


         Output: - out_lst(type, list, mandatory):
                   by default the output arguments are passed with a list.
                   In this case, out is a list containing the MFCCs.
                   The tensor is formatted in one of the following ways
                   depending on the input shape: [batch, n_mfcc, time_steps],
                   [batch, channels, n_mfcc, time_steps]
        """
        # Computing MFCCs by applying the DCT transform
        mfcc = torch.matmul(
            fbanks.transpose(1, -1), self.dct_mat.to(fbanks.device)
        ).transpose(1, -1)

        return mfcc


class deltas(torch.nn.Module):
    """
    -------------------------------------------------------------------------
    data_processing.deltas (author: Mirco Ravanelli)

    Description:  This class computes time derivatives of an input tensor.
                  It is a modification of the spectrogram of the torch audio
                  toolkit (https://github.com/pytorch/audio).

    Input: - der_win_length (type: int(3,inf), default: 3):
              it the length of the window used to compute the
              time derivatives.

    Example:  import torch
              from data_processing import deltas

              # Initialization of the delta class
              compute_deltas=deltas()

              # Delta computations
              batch=torch.rand([4,13,230])
              delta_out=compute_deltas(batch)
              print(delta_out)
              print(delta_out[0].shape)
    -------------------------------------------------------------------------
    """

    def __init__(
        self,
        der_win_length=5,
    ):
        super().__init__()
        self.der_win_length = der_win_length

        def hook(self, input):
            self.kernel = self.kernel.repeat(input[0].shape[-2], 1, 1)
            self.hook.remove()

        self.hook = self.register_forward_pre_hook(hook)

        # Additional parameters
        self.n = (self.der_win_length - 1) // 2
        self.denom = self.n * (self.n + 1) * (2 * self.n + 1) / 3
        self.kernel = torch.arange(-self.n, self.n + 1, 1).float()

    def forward(self, x):
        """
        Input: - x (type, torch.Tensor, mandatory):
                   The input tensor must be 2-4 dims, with time_steps last.

        Output: deltas (type, torch.Tensor):
                   The same shape as the inputs
        """
        # Managing multi-channel deltas reshape tensor (batch*channel,time)
        or_shape = x.shape

        if len(or_shape) == 4:
            x = x.view(or_shape[0] * or_shape[2], or_shape[1], or_shape[3])

        # Doing padding for time borders
        x = torch.nn.functional.pad(x, (self.n, self.n), mode="replicate")

        # Derivative estimation (with a fixed convolutional kernel)
        delta_coeff = (
            torch.nn.functional.conv1d(
                x, self.kernel.to(x.device), groups=x.shape[1]
            )
            / self.denom
        )

        # Retrieving the original dimensionality (for multi-channel case)
        if len(or_shape) == 4:
            delta_coeff = delta_coeff.reshape(
                or_shape[0],
                or_shape[1],
                delta_coeff.shape[1],
                delta_coeff.shape[2],
            )

        return delta_coeff


class context_window(torch.nn.Module):
    """
     -------------------------------------------------------------------------
     data_processing.context_window (author: Mirco Ravanelli)

     Description:  This class applies a context window by gathering multiple
                   time steps in a single feature vector. The operation is
                   performed with a convolutional layer based on a fixed
                   kernel designed for that.

     Input: - left_frames (type: int(0,inf), default: 0):
               it is this the number of left frames (i.e, past
               frames to collect)

           - right_frames (type: int(0,inf), default: 0):
               it is this the number of right frames (i.e,
               future frames to collect)
     Example:  import torch
               from data_processing import context_window

               # delta config dictionary definition
               config={'class_name':'data_processing.context_window',
                       'left_frames': '5',
                       'right_frames': '5'}

               # Initialization of the delta class
               compute_cw=context_window(config,
                                         first_input=[torch.rand([4,13,100])])

               # context computations
               batch=torch.rand([4,13,230])
               batch_cw=compute_cw([batch])
               print(batch_cw)
               print(batch_cw[0].shape)
     -------------------------------------------------------------------------
     """

    def __init__(
        self,
        left_frames=0,
        right_frames=0,
    ):
        super().__init__()

        self.left_frames = left_frames
        self.right_frames = right_frames

        # Additional parameters
        self.context_len = self.left_frames + self.right_frames + 1
        self.kernel_len = 2 * max(self.left_frames, self.right_frames) + 1

        # Kernel definition
        self.kernel = torch.eye(self.context_len, self.kernel_len)

        if self.right_frames > self.left_frames:
            lag = self.right_frames - self.left_frames
            self.kernel = torch.roll(self.kernel, lag, 1)

        self.first_call = True

    def forward(self, x):
        """
        Input: - x (type: torch.Tensor, mandatory):
                   The input tensor must be 2-4 dims with time_steps last.

        Output: tensor with context (type: torch.Tensor)
                    Tensor has context added, the second from the end
                    dimension has been increased in size.
        """

        if self.first_call is True:
            self.first_call = False
            self.kernel = (
                self.kernel.repeat(x.shape[1], 1, 1)
                .view(
                    x.shape[1] * self.context_len,
                    self.kernel_len,
                )
                .unsqueeze(1)
            )

        # Managing multi-channel case
        or_shape = x.shape

        if len(or_shape) == 4:
            x = x.reshape(or_shape[0] * or_shape[2], or_shape[1], or_shape[3])

        if len(or_shape) == 5:
            x = x.reshape(
                or_shape[0] * or_shape[2] * or_shape[3],
                or_shape[1],
                or_shape[4],
            )

        # Compute context (using the estimated convolutional kernel)
        cw_x = torch.nn.functional.conv1d(
            x,
            self.kernel.to(x.device),
            groups=x.shape[1],
            padding=max(self.left_frames, self.right_frames),
        )

        # Retrieving the original dimensionality (for multi-channel case)
        if len(or_shape) == 4:
            cw_x = cw_x.reshape(
                or_shape[0], cw_x.shape[1], or_shape[2], cw_x.shape[-1]
            )

        if len(or_shape) == 5:
            cw_x = cw_x.reshape(
                or_shape[0],
                cw_x.shape[1],
                or_shape[2],
                or_shape[3],
                cw_x.shape[-1],
            )

        return cw_x


class mean_var_norm(torch.nn.Module):
    def __init__(
        self,
        mean_norm=True,
        std_norm=True,
        norm_type='global',
        avg_factor=None,
        do_recovery=True,
        output_folder=None,
        requires_grad=False,
    ):
        super().__init__()

        self.mean_norm = mean_norm
        self.std_norm = std_norm
        self.norm_type = norm_type
        self.avg_factor = avg_factor
        self.recovery = do_recovery
        self.output_folder = output_folder
        self.requires_grad = requires_grad

        # Parameter initialization
        self.glob_mean = 0
        self.glob_std = 0
        self.spk_dict_mean = {}
        self.spk_dict_std = {}
        self.spk_dict_count = {}
        # Counter initialization
        self.count = 0

        # Numerical stability for std
        self.eps = 1e-10

        # Recovery stored stats
        # recovery(self)

    def forward(self, x, lengths, spk_ids=None):

        self.device_inp = str(x.device)

        N_batches = x.shape[0]

        current_means = []
        current_stds = []

        x = x.unsqueeze(1).transpose(1, -1).squeeze(-1)

        for snt_id in range(N_batches):

            # Avoiding padded time steps
            actual_size = int(torch.round(lengths[snt_id] * x.shape[1]))

            # computing statistics
            current_mean, current_std = self.compute_current_stats(
                x[snt_id, 0:actual_size, ...], (0)
            )

            # appending values
            current_means.append(current_mean)
            current_stds.append(current_std)

            # sentence normalization
            if self.norm_type == "sentence":

                x[snt_id] = (x[snt_id] - current_mean.data) / current_std.data

            if self.norm_type == "speaker":

                spk_id = int(spk_ids[snt_id][0])

                if spk_id not in self.spk_dict_mean:

                    # Initialization of the dictionary
                    self.spk_dict_mean[spk_id] = current_mean
                    self.spk_dict_std[spk_id] = current_std
                    self.spk_dict_count[spk_id] = 1

                else:
                    self.spk_dict_count[spk_id] = (
                        self.spk_dict_count[spk_id] + 1
                    )

                    if self.avg_factor is None:
                        self.weight = 1 / self.spk_dict_count[spk_id]
                    else:
                        self.weight = self.avg_factor

                    self.spk_dict_mean[spk_id] = (
                        1 - self.weight
                    ) * self.spk_dict_mean[spk_id] + self.weight * current_mean
                    self.spk_dict_std[spk_id] = (
                        1 - self.weight
                    ) * self.spk_dict_std[spk_id] + self.weight * current_std

                    self.spk_dict_mean[spk_id].detach()
                    self.spk_dict_std[spk_id].detach()

                x[snt_id] = (
                    x[snt_id] - self.spk_dict_mean[spk_id].data
                ) / self.spk_dict_std[spk_id].data

        if self.norm_type == "batch" or self.norm_type == "global":

            current_mean = torch.mean(torch.stack(current_means), dim=0)
            current_std = torch.mean(torch.stack(current_stds), dim=0)

            if self.norm_type == "batch":
                x = (x - current_mean.data) / (current_std.data)

            if self.norm_type == "global":

                if self.count == 0:
                    self.glob_mean = current_mean
                    self.glob_std = current_std

                else:
                    if self.avg_factor is None:
                        self.weight = 1 / (self.count + 1)
                    else:
                        self.weight = self.avg_factor

                    self.glob_mean = (
                        1 - self.weight
                    ) * self.glob_mean + self.weight * current_mean

                    self.glob_std = (
                        1 - self.weight
                    ) * self.glob_std + self.weight * current_std

                self.glob_mean.detach()
                self.glob_std.detach()

                x = (x - self.glob_mean.data) / (self.glob_std.data)

        # Update counter
        self.count = self.count + 1

        x = x.unsqueeze(-1).transpose(1, -1).squeeze(1)

        return x

    def compute_current_stats(self, x, dims):

        # Compute current mean
        if self.mean_norm:
            current_mean = torch.mean(x, dim=dims).detach().data
        else:
            current_mean = torch.tensor([0.0]).to(x.device)

        # Compute current std
        if self.std_norm:
            current_std = torch.std(x, dim=dims).detach().data
        else:
            current_std = torch.tensor([1.0]).to(x.device)

        # Improving numerical stability of std
        current_std = torch.max(
            current_std, self.eps * torch.ones_like(current_std)
        )

        return current_mean, current_std

    def statistics_dict(self):

        state = {}
        state["count"] = self.count
        state["glob_mean"] = self.glob_mean
        state["glob_std"] = self.glob_std
        state["spk_dict_mean"] = self.spk_dict_mean
        state["spk_dict_std"] = self.spk_dict_std
        state["spk_dict_count"] = self.spk_dict_count

        return state

    def load_statistics_dict(self, state):

        self.count = state["count"]
        if isinstance(state["glob_mean"], int):
            self.glob_mean = state["glob_mean"]
            self.glob_std = state["glob_std"]
        else:
            self.glob_mean = state["glob_mean"]  # .to(self.device_inp)
            self.glob_std = state["glob_std"]  # .to(self.device_inp)

        # Loading the spk_dict_mean in the right device
        self.spk_dict_mean = {}
        for spk in state["spk_dict_mean"]:
            self.spk_dict_mean[spk] = state["spk_dict_mean"][spk].to(
                self.device_inp
            )

        # Loading the spk_dict_std in the right device
        self.spk_dict_std = {}
        for spk in state["spk_dict_std"]:
            self.spk_dict_std[spk] = state["spk_dict_std"][spk].to(
                self.device_inp
            )

        self.spk_dict_count = state["spk_dict_count"]

        return state
