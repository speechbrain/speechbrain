"""This lobe enables the integration of huggingface pretrained LLAMA2-chat model.

Transformer from HuggingFace needs to be installed:
https://huggingface.co/transformers/installation.html

Authors
 * Pooneh Mousavi 2023
"""

import logging
from torch import Tensor
import torch
import torch.nn as nn
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from speechbrain.lobes.models.huggingface_transformers.huggingface_new import (
    HFTransformersInterface,
)
from transformers import BitsAndBytesConfig


logger = logging.getLogger(__name__)


class HuggingFaceLLAMA2(HFTransformersInterface):
    """This lobe enables the integration of HuggingFace pretrained LLAMA2 model.
     Source paper LLAMA2:
       https://arxiv.org/abs/2307.09288
    Transformer from HuggingFace needs to be installed:
        https://huggingface.co/transformers/installation.html

    The model can be finetuned. It will download automatically the model from
    HuggingFace or use a local path.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "meta-llama/Llama-2-7b-chat-hf"
    save_path : str
        Path (dir) of the downloaded model.
    freeze : bool (default: False)
        If True, the model is frozen. If False, the model will be trained
        alongside with the rest of the pipeline.
    Example
    -------
    >>> model_hub = "meta-llama/Llama-2-7b-chat-hf"
    >>> save_path = "savedir"
    >>> model = HuggingFaceLLAMA2(model_hub, save_path)
    >>> tokens = torch.tensor([[1, 1]])
    >>> attention_mask = torch.tensor([[1, 1]])
    >>> outputs = model(tokens, tokens_type, attention_mask)
    """

    def __init__(
        self,
        source: str,
        save_path: str,
        freeze: bool = False,
        max_new_tokens: int = 200,
        use_4bit: bool = True,
        bnb_4bit_compute_dtype: str = "float16",
        bnb_4bit_quant_type: str = "nf4",
        use_nested_quant: bool = False,
        min_length: int = 1,
        top_k: int = 45,
        top_p: float = 0.9,
        num_beams: int = 8,
        early_stopping: bool = True,
        with_peft : bool = False,
    ) -> None:

        
        self.max_new_tokens = max_new_tokens
        self.min_length = min_length
        self.top_k = top_k
        self.top_p = top_p
        self.num_beams = num_beams
        self.early_stopping = early_stopping

        compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
        bnb_config = None
        if with_peft:                
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=use_4bit,
                bnb_4bit_quant_type=bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=use_nested_quant,
        )
            # Check GPU compatibility with bfloat16
            if compute_dtype == torch.float16 and use_4bit:
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    logger.info("=" * 80)
                    logger.info(
                        "Your GPU supports bfloat16: accelerate training with bf16=True"
                    )
                    logger.info("=" * 80)

        super().__init__(
            source=source, save_path=save_path, freeze=freeze, with_casual_lm=True, quantization_config=bnb_config
        )

        self.load_tokenizer(source=source, pad_token=None, use_fast=False)
        # Define a custom padding token
        self.tokenizer.pad_token = "<PAD>"
        # Set the padding direction to the right
        self.tokenizer.padding_side = "right"


        if with_peft:    
            self.model = prepare_model_for_kbit_training(self.model)

            config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.1,
                r=64,
                bias="none",
                task_type="CAUSAL_LM",
            )

            self.model = get_peft_model(self.model, config)
            self.print_trainable_parameters(self.model)





    def forward(
        # self, input_ids: Tensor
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ):
        """ Takes an input a history of conversation and returns its corresponding reply.

        Arguments
        ---------
        input_ids : torch.Tensor ()
            A batch of input-id to transform to features.
        attention_mask : torch.Tensor ()
            A batch of attention_mask.
        """
        with torch.set_grad_enabled(not self.freeze):
            output = self.model.forward(
                input_ids, attention_mask=attention_mask,
            )
        return output

    def generate(
        self, input_ids: Tensor, attention_mask: Tensor, decoder_type="greedy",
    ):
        """ Takes an input a history of conversation and returns its corresponding reply.

        Arguments
        --------
        input_ids : torch.Tensor ()
            A batch of input-id   which are dialogue context tokens
        # decoder_type : Str
        #     It shows strategy for autoregressive decoding either beam seach or greedy.
        # attention_mask : torch.Tensor ()
        #     A batch of attention_mask.
        """

        with torch.no_grad():
            if decoder_type == "beam":
                # beam decoding based on the input_ids which are dialogue context tokens (here only history)
                hyp = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    max_new_tokens=self.max_new_tokens,
                    min_length=self.min_length,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    temperature=1.0,
                    num_beams=self.num_beams,
                    num_return_sequences=1,
                    repetition_penalty=1.0,
                    length_penalty=1,
                    early_stopping=self.early_stopping,
                )
            else:
                # greedy decoding based on the input_ids which are dialogue context tokens (here only history)
                hyp = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.max_new_tokens,
                    # pad_token_id=50258,
                    # eos_token_id=50258,
                    attention_mask=attention_mask,
                )
        return hyp

    def print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logger.info(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )


model_hub = "meta-llama/Llama-2-7b-chat-hf"
save_path = "savedir"
model = HuggingFaceLLAMA2(model_hub, save_path, with_peft=True)
tokens = torch.tensor([[1, 1]])
attention_mask = torch.tensor([[1, 1]])
outputs = model(tokens, attention_mask)