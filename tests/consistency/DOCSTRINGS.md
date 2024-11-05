# Docstrings in SpeechBrain
All the functions or classes of SpeechBrain must have a docstring. SpeechBrain adopts the NumPy-like style for the docstrings.
- Here is an example of a class:

>     class SincConv(nn.Module):
>     	 """This function implements SincConv (SincNet).
>
>     	 M. Ravanelli, Y. Bengio, "Speaker Recognition from raw waveform with
>     	 SincNet", in Proc. of SLT 2018 (https://arxiv.org/abs/1808.00158)
>
>     	 Arguments
>     	 ---------
>     	 input_shape : tuple
>     	 The shape of the input. Alternatively use ``in_channels``.
>     	 in_channels : int
>     	 The number of input channels. Alternatively use ``input_shape``.
>     	 out_channels : int
>     	 It is the number of output channels.
>     	 kernel_size: int
>     	 Kernel size of the convolutional filters.
>     	 stride : int
>     	 Stride factor of the convolutional filters. When the stride factor > 1,
>     	 a decimation in time is performed.
>     	 dilation : int
>     	 Dilation factor of the convolutional filters.
>     	 padding : str
>     	 (same, valid, causal). If "valid", no padding is performed.
>     	 If "same" and stride is 1, the output shape is the same as the input shape.
>     	 "causal" results in causal (dilated) convolutions.
>     	 padding_mode : str
>     	 This flag specifies the type of padding. See torch.nn documentation
>     	 for more information.
>     	 groups: int
>     	 This option specifies the convolutional groups. See torch.nn
>     	 documentation for more information.
>     	 bias : bool
>     	 If True, the additive bias b is adopted.
>     	 sample_rate : int,
>     	 The sampling rate of the input signals. It is only used for sinc_conv.
>     	 min_low_hz : float
>     	 Lowest possible frequency (in Hz) for a filter. It is only used for
>     	 sinc_conv.
>     	 min_low_hz : float
>     	 Lowest possible value (in Hz) for a filter bandwidth.
>
>     	 Example
>     	 -------
>     	 >>> inp_tensor = torch.rand([10, 16000])
>     	 >>> conv = SincConv(input_shape=inp_tensor.shape, out_channels=25, kernel_size=11)
>     	 >>> out_tensor = conv(inp_tensor)
>     	 >>> out_tensor.shape
>     	 torch.Size([10, 16000, 25])
>     	 """

Here is an example of a function:

>     def ngram_perplexity(eval_details, logbase=10.0):
>     	 """
>     	 Computes perplexity from a list of individual sentence evaluations.
>
>     	 Arguments
>     	 ---------
>     	 eval_details : list
>     	 List of individual sentence evaluations. As returned by
>     	 `ngram_evaluation_details`
>     	 logbase : float
>     	 The logarithm base to use.
>
>     	 Returns
>     	 -------
>     	 float
>     	 The computed perplexity.
>
>     	 Example
>     	 -------
>     	 >>> eval_details = [
>     	 ... collections.Counter(neglogprob=5, num_tokens=5),
>     	 ... collections.Counter(neglogprob=15, num_tokens=15)]
>     	 >>> ngram_perplexity(eval_details)
>     	 10.0

We strongly encourage contributors to add a runnable example for the most important functions and classes. The examples will be tested automatically with pytest and help clarify how the function/classes should be used. We also encourage contributors to accurately describe the arguments and returns of a function (along with their types).  Short docstring (e.g., 1-line) are acceptable for minor functions only, but we encourage anyway to describe at least the inputs and outputs.
