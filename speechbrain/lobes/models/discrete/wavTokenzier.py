import torch
from wavtokenizer import WavTokenizer 
from wavtokenizer.encoder.utils import convert_audio
import torchaudio
import torch
from wavtokenizer.decoder.pretrained import WavTokenizer


device=torch.device('cpu')

config_path = "wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml"
model_path = "WavTokenizer_small_320_24k_4096.ckpt"
audio_outpath = "savedir"

wavtokenizer = WavTokenizer.from_pretrained0802(config_path, model_path)

wav = torch.randn(1, 24000)
bandwidth_id = torch.tensor([0])
with torch.no_grad():
    features, codes = wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id)
    print(codes.shape)
    audio_out = wavtokenizer.decode(features, bandwidth_id=bandwidth_id)
    print(audio_out.shape)