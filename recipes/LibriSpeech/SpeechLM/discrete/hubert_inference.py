"""
HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 python test.py hparams/ssl.yaml  --data_folder $SLURM_TMPDIR/LibriSpeech/ --tokens_folder $SCRATCH/results/dac/librispeech/  --num_workers 4 --num_codebooks 1  --eval_precision=bf16 --batch_size=16 --block_size=2048 --grad_accumulation_factor=8 --max_grad_norm=1.0 --optimizer_step_limit 10_000 --number_of_epochs=500 --tqdm_colored_bar
"""
from speechbrain.utils.checkpoints import Checkpointer
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import sys
import torch
import torchaudio 
from utils.tokenizer_interface import FairseqHuBERTTokenizer
from textless.vocoders.hifigan.vocoder import CodeHiFiGANVocoder

hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin, overrides)

prompt_audio = "/home/adelmou/proj/speechbrain/gslms/speechbrain/OVBUlHqrvK.wav"
checkpoint_dir = "/scratch/adelmou/results/speech_lm/hubert25hzl11_collapsed//save/"

tokenizer = FairseqHuBERTTokenizer(
    layer=11,
    feat_extractor_path="/scratch/adelmou/models/hubert25hz/mhubert_base_25hz_cp_mls_cv_sp_fisher.pt",
    km_path="/scratch/adelmou/models/hubert25hz/mhubert_base_25hz_cp_mls_cv_sp_fisher_L11_km500.bin"
)
tokenizer.to("cuda").eval()

vocab_size = 500
vocoder = CodeHiFiGANVocoder.by_name(
    dense_model_name = "mhubert-base-25hz",
    quantizer_model_name = "kmeans", 
    vocab_size = vocab_size
).eval().to("cuda")

x, _ = torchaudio.load(prompt_audio)
x = x.to("cuda").view(1, 1, -1)

tokens = tokenizer.sig_to_tokens(x, None)
print(tokens.shape)
tokens = tokens.permute(0, 2, 1)
gen_wav = vocoder(tokens, dur_prediction = True)

# Ensure `gen_wav` is on CPU and convert to the appropriate dtype if necessary
gen_wav_cpu = gen_wav.unsqueeze(0).cpu()

# Save the first audio in the batch to 'output.wav'
torchaudio.save("prompt_audio.wav", gen_wav_cpu, sample_rate=16_000)
exit()
# goal: load model and do a generation.
ckpt_finder = Checkpointer(checkpoint_dir)
best_ckpt = ckpt_finder.find_checkpoint(min_key="loss")
model_ckpt = best_ckpt.paramfiles["model"]
model = hparams["model"].to("cuda").to(torch.bfloat16)
sb.utils.checkpoints.torch_parameter_transfer(model, model_ckpt)

# audio_tokens = torch.full([1, 1, 8], hparams["eos_token"]).to("cuda").to(torch.int64)

from torch.nn import functional as F

def sample_from_logits(logits, temperature=1.0, top_k=None):
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[..., [-1]]] = -float('Inf')
    # apply softmax to convert logits to (normalized) probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution
    return torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)

temperature = 1.0
top_k = 100

def generate(prompt, max_new_tokens):
    B, K, T = prompt.shape
    first_param = next(iter(model.parameters()))
    device = first_param.device
    audio_preds_masks = torch.ones((B, T), dtype=bool, device=prompt.device)
    max_gen_len = T + max_new_tokens
    assert max_gen_len <= hparams["block_size"]
    # this token is used as default value for codes that are not generated yet
    unknown_token = -1
    # we generate codes up to the max_gen_len that will be mapped to the pattern sequence
    gen_codes = torch.full((B, K, max_gen_len), unknown_token, dtype=torch.long, device=device)
    # filling the gen_codes with the prompt if needed
    gen_codes[..., :T] = prompt
    audio_tokens = prompt[..., :-1].clone()

    input_audio_tokens = hparams["codebook_pattern"].apply_delay_pattern(audio_tokens)
    # compute the frame audio embeddings as the sum of codebook embeddings
    h = sum([hparams["model"].audio_in_embds[k](input_audio_tokens[:, k]) for k in range(hparams["num_codebooks"])])
    # generation loop
    for offset in range(prompt.size(2), max_gen_len):
        gen_iter = offset - prompt.size(2)
        tokens = gen_codes[..., [offset - 1]]
        h = sum([hparams["model"].audio_in_embds[k](tokens[:, k]) for k in range(hparams["num_codebooks"])])
        out = hparams["model"].model.model(inputs_embeds=h, use_cache=False)
        h_ctx = out['last_hidden_state']
        logits = torch.stack([
            hparams["model"].audio_out[k](h_ctx[:, [-1], :]) for k in range(hparams["num_codebooks"])
        ], dim=1)
        logits = logits[:, :, -1, :] # crop to just the final time step
        logits[..., -2:] = -float('Inf') # forbid generating special tokens
        idx_next = sample_from_logits(logits, temperature, top_k).reshape(*list(logits.shape[:-1]))
        if gen_iter < (hparams["num_codebooks"] - 1):
            idx_next[:, gen_iter + 1:] = hparams["pad_token"]
        gen_codes[..., offset] = idx_next

    if audio_preds_masks[..., prompt.size(2) - 1]: # remove delay from generated audio
        generated_chunk = gen_codes[..., T:]
        unpadded_length = generated_chunk.size(2) - hparams["num_codebooks"]
        undelayed_sequence = torch.full_like(generated_chunk, hparams["pad_token"], dtype=generated_chunk.dtype, device=generated_chunk.device)
        # Reconstruct the original sequence by removing the delays
        for i in range(hparams["num_codebooks"]):
            undelayed_sequence[:, i, :-hparams["num_codebooks"]] = generated_chunk[:, i, i:i + unpadded_length]
        gen_codes = undelayed_sequence[..., :-hparams["num_codebooks"]]
    else:
        gen_codes = gen_codes[:, 0, T:].view(-1)
    
    return gen_codes

ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
with ctx:
    gen_codes = generate(prompt_audio, 512)
with torch.no_grad():

    if audio_tokenizer == "speech-tokenizer":
        gen_wav = codec.decode(gen_codes.permute(1, 0, 2))
    else:
        z_bdt = codec.quantizer.from_codes(gen_codes)[0]
        gen_wav = codec.decode(z_bdt)


# Define the sample rate. Replace `audio_sr` with your actual sample rate variable.
audio_sr = 16000  # Example sample rate (16kHz). Update as needed.

# Ensure `gen_wav` is on CPU and convert to the appropriate dtype if necessary
gen_wav_cpu = gen_wav[0].cpu()

# Save the first audio in the batch to 'output.wav'
torchaudio.save("output.wav", gen_wav_cpu, sample_rate=audio_sr)
