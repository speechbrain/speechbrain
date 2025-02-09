import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
import sys
import torch
import torchaudio 
from torch.nn import functional as F


hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

with open(hparams_file) as fin:
    hparams = load_hyperpyyaml(fin, overrides)
    
# print(hparams)
tokenizer = hparams["tokenizer"].to("cuda").eval()

import os 
os.makedirs("samples", exist_ok=True)

x, _ = torchaudio.load(hparams["prompt_audio_file"])
x = x.to("cuda").view(1, -1)

if hparams["prompt_duration"] != -1:
    assert hparams["prompt_duration"] > 0
    num_samples = int(hparams["prompt_duration"] * hparams["sample_rate"])
    # before cropping, synthesize the original audio
    init_wav = tokenizer.tokens_to_sig(tokenizer.sig_to_tokens(x, None))
    # save it
    torchaudio.save(f"samples/{hparams['prefix_output']}_target.wav", init_wav.cpu(), hparams["sample_rate"])
    x = x[:, :num_samples]

prompt_audio_tokens = tokenizer.sig_to_tokens(x, None)

print(prompt_audio_tokens.shape)

gen_wav = tokenizer.tokens_to_sig(prompt_audio_tokens)

print(gen_wav.shape)

torchaudio.save(f"samples/{hparams['prefix_output']}_prompt_audio.wav", gen_wav.cpu(), hparams["sample_rate"])

model_ckpt = hparams["checkpointer"].find_checkpoint(min_key="loss").paramfiles["model"]
model = hparams["model"].to("cuda").to(torch.bfloat16)
sb.utils.checkpoints.torch_parameter_transfer(model, model_ckpt)

def sample_from_logits(logits, temperature=1.0, top_k=None):
    logits = logits / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[..., [-1]]] = -float('Inf')
    # apply softmax to convert logits to (normalized) probabilities
    probs = F.softmax(logits, dim=-1)
    # sample from the distribution
    return torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1)

temperature = 1.5
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
    prompt_audio_tokens = prompt_audio_tokens.permute(0, 2, 1)
    gen_codes = generate(prompt_audio_tokens, 100)

gen_codes = gen_codes.permute(0, 2, 1)
gen_wav = tokenizer.tokens_to_sig(gen_codes)
torchaudio.save(f"samples/{hparams['prefix_output']}_generated_audio.wav", gen_wav.cpu(), hparams["sample_rate"])



