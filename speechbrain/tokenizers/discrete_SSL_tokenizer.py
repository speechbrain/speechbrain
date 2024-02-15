import numpy as np
import torch
class DiscreteSSLTokenizer:
    """This lobe enables the integration of HuggingFace and SpeechBrain
    pretrained Discrete HuBERT models.

    Source paper HuBERT: https://arxiv.org/abs/2106.07447
    Transformer from HuggingFace needs to be installed:
    https://huggingface.co/transformers/installation.html

    The model can be used as a fixed Discrete feature extractor or can be finetuned. It
    will download automatically the model from HuggingFace or use a local path.

    For now, HuggingFace's HuBERT and WavLM model can be loaded using the exact code for Wav2Vec2 model.
    For this reason, HuBERT and WavLM can be fine inheriting the Wav2Vec2 class.

    Arguments
    ---------
    source : str
        HuggingFace hub name: e.g "facebook/hubert-base-ls960"
    save_path : str
        Path (dir) of the downloaded model.

    Example
    -------
    >>> import torch
    >>> inputs = torch.rand([10, 2000])
    >>> model_hub = "facebook/hubert-base-ls960"
    >>> save_path = "savedir"
    >>> ssl_layer_num = [5,7]
    >>> kmeans_repo_id = "speechbrain/SSL_Quantization"
    >>> kmeans_dataset = "LibriSpeech-100-360-500"
    >>> kmeans_cache_dir="savedir"
    >>> num_clusters = 128

    >>> model = DiscreteHuBERT(model_hub, save_path,freeze = True,kmeans_repo_id=kmeans_repo_id, kmeans_dataset=kmeans_dataset, kmeans_cache_dir=kmeans_cache_dir,num_clusters=num_clusters)
    >>> embs, tokens = model(inputs,ssl_layer_num=ssl_layer_num,deduplicte=True)
    >>> print(embs.shape)
    torch.Size([10, 4, 2])
    >>> print(tokens.shape)
    torch.Size([10, 4, 2, 768])
    """

    def __init__(self,num_clusters):
        self.num_clusters = num_clusters
    
    def textify(self,tokens,layer):
        tokens_char =[]
        # tokens = [row - layer *  self.num_clusters for row in input]
        for row in tokens:
            tokens_char.append(" ".join( [chr((token- layer *  self.num_clusters) + 97) for token in row]))
        return tokens_char

    def encode(self, input,  SSL_layers=[7], deduplicates=[False], bpe_tokenizers=[None]):
        assert input.shape[2] == len(SSL_layers), f"input shape:{input.shape} has conflicts with the length of provided SSL_layers: {len(SSL_layers)}. The second dimention of input should be the same  as number of layers!!!"
        assert len(deduplicates) == len(SSL_layers) == len(bpe_tokenizers), f"length of SSL_layers,deduplicates,bpe_tokenizers should be the same!!!"
        # add 1 to all cluster ids to avoid conflict wih pad_value==0
        input = input + 1
        token_ids = []
        for i,duplicate in enumerate(deduplicates):
            tokens= []
            if duplicate:
                unique_token_ids = [
                        row[np.diff(row, prepend=np.nan).astype(bool)]
                        for row in input[:,:,i]
                    ]
                layer_token_ids = [
                        torch.tensor(
                            row ,
                            dtype=torch.long,
                            device=input.device,
                        )
                        for row in unique_token_ids
                    ]
                tokens.extend(layer_token_ids)

            else:
                tokens.extend(input[:,:,i])
            
            if bpe_tokenizers[i] != " ":
                token_char = self.textify(tokens,SSL_layers[i])
                token_ids.extend([bpe_tokenizers[i].encode_as_ids(char) for char in token_char]) + SSL_layers[i] *  self.num_clusters
            else:
                 token_ids.extend(tokens)
                

        return torch.stack(
                torch.split(
                    torch.nn.utils.rnn.pad_sequence(
                        token_ids, batch_first=True
                    ),
                    3,
                ),
                dim=2,
            )
        
        # token_ids = []
        # for i in range(len(bpe_tokenizers)):
        #     if bpe_tokenizers:
        #         pass
        #     else:
        #         token_ids.extend(input[:,:,i])

        







#         pass
# import torch
# tokenizer= DiscreteSSLTokenizer(128)
# input= torch.tensor([[[760, 984],
#          [672, 984],
#          [672, 984],
#          [672, 984],
#          [672, 984],
#          [672, 984]],

#         [[672, 984],
#          [672, 984],
#          [672, 984],
#          [672, 984],
#          [672, 984],
#          [689, 984]],

#         [[672, 984],
#          [760, 984],
#          [672, 984],
#          [672, 984],
#          [672, 984],
#          [672, 984]]])
# output= tokenizer.encode(input,[5,7],[False,True],[None,None])
# print(output.shape)