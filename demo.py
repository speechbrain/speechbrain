from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import gradio as gr

model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')

def speechbrain(aud):
  est_sources = model.separate_file(path=aud.name) 
  torchaudio.save("source1hat.wav", est_sources[:, :, 0].detach().cpu(), 8000)
  torchaudio.save("source2hat.wav", est_sources[:, :, 1].detach().cpu(), 8000)
  return "source1hat.wav", "source2hat.wav"

inputs = gr.inputs.Audio(label="Input Audio", type="file")
outputs =  [
  gr.outputs.Audio(label="Output Audio One", type="file"),
  gr.outputs.Audio(label="Output Audio Two", type="file")
]

title = "Speech Seperation"
description = "demo for Speech Seperation by SpeechBrain. To use it, simply upload your audio, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2010.13154'>Attention is All You Need in Speech Separation</a> | <a href='https://github.com/speechbrain/speechbrain/tree/develop/recipes/WSJ0Mix/separation'>Github Repo</a></p>"
examples = [
    ['samples/audio_samples/test_mixture.wav']
]
gr.Interface(speechbrain, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()