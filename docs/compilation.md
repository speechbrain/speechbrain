#  Compilation

Compilation of your models in SpeechBrain can potentially improve their speed and reduce memory demand. SpeechBrain inherits the compilation methods supported by PyTorch, including the just-in-time compiler (JIT) and the `torch.compile` method introduced in PyTorch version >=2.0.

## Compile with `torch.compile`
The `torch.compile` feature was introduced with PyTorch version >=2.0 to gradually replace JIT. Although this feature is valuable, it is still in the beta phase, and improvements are ongoing. Please have a look at the [PyTorch documentation](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) for more information.

### How to use `torch.compile`
Compiling all modules in SpeechBrain is straightforward. You can enable compilation by using the `--compile` flag in the command line when running a training recipe. For example:

```bash
python train.py train.yaml --data_folder=your/data/folder --compile
```

This will automatically compile all the modules declared in the YAML file under the `modules` section.

Note that you might need to configure additional compilation flags correctly (e.g., `--compile_mode`, `--compile_using_fullgraph`, `--compile_using_dynamic_shape_tracing`) to ensure successful model compilation or achieve the best performance. For a deeper understanding of their roles, refer to the documentation in the [PyTorch documentation](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html).

In some cases, you may want to compile only specific modules. To achieve this, add a list of the module keys you want to compile in the YAML file using `compile_module_keys`. For instance:

```yaml
compile_module_keys: [encoder, decoder]
```

This will compile only the encoder and decoder models, which should be declared in the YAML file before using the respective keys.

Remember to call the training script with the `--compile` flag.

**Note of caution**: Compiling a model can be a complex process and may take some time. Additionally, it may fail in certain cases. The speed-up achieved through compilation is highly dependent on the system and GPU being used. For example, higher-end GPUs like the A100 tend to yield better speed-ups, while you may not observe significant improvements with V100 GPUs. We support this feature with the hope that `torch.compile` will constantly improve over time.

## Compile with JIT
JIT was the first compilation method supported by PyTorch. It is important to note that JIT is expected to be replaced soon by `torch.compile`. Please have a look at the [PyTorch documentation](https://pytorch.org/docs/stable/jit.html) for more information.

### How to use JIT
To compile all modules in SpeechBrain using JIT, use the `--jit` flag in the command line when running a training recipe:

```bash
python train.py train.yaml --data_folder=your/data/folder --jit
```

This will automatically compile all the modules declared in the YAML file under the `modules` section.

If you only want to compile specific modules, add a list of the module keys you want to compile in the YAML file using `jit_module_keys`. For example:

```yaml
jit_module_keys: [encoder, decoder]
```
This will compile only the encoder and decoder models, provided they are declared in the YAML file using the specified keys.

Remember to call the training script with the `--jit` flag.

**Note of caution**: JIT has specific requirements for supported syntax, and many popular Python syntaxes are not supported. Therefore, when designing a model with JIT in mind, ensure that it meets the necessary syntax requirements for successful compilation. Additionally, the speed-up achieved through JIT compilation varies depending on the model type. We found it most beneficial for custom RNNs, such as the Li-GRU used in SpeechBrain's TIMIT/ASR/CTC. Custom RNNs often require "for loops," which can be slow in Python. The compilation with JIT provides a significant speed-up in such cases.

