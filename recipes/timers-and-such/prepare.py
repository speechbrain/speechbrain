import pandas as pd
from speechbrain.data_io.data_io import read_wav_soundfile
import ast

folder = "/home/mila/l/lugoschl/data/timers-and-such/"
type = 1
splits = ["train-real", "dev-real", "test-real", "train-synth", "dev-synth", "test-synth"]
for split in splits:
	ID = []
	duration = []

	wav = []
	wav_format = []
	wav_opts = []

	spk_id = []
	spk_id_format = []
	spk_id_opts = []

	semantics = []
	semantics_format = []
	semantics_opts = []

	transcript = []
	transcript_format = []
	transcript_opts = []

	df = pd.read_csv(folder + split + ".csv")
	for i in range(len(df)):
		ID.append(i)
		signal = read_wav_soundfile(folder + df.path[i])
		duration.append(signal.shape[0] / 16000)

		wav.append("/localscratch/timers-and-such/" + df.path[i])
		wav_format.append("wav")
		wav_opts.append(None)

		spk_id.append(df.speakerId[i])
		spk_id_format.append("string")
		spk_id_opts.append(None)

		if type == 1:
			semantics.append(df.semantics[i].replace(",", "|")) # Commas in dict will make using csv files tricky; replace with pipe.
		if type == 2:
			dict = ast.literal_eval()
			semantics.append()
		if type == 3:
			semantics.append()
		semantics_format.append("string")
		semantics_opts.append(None)

		transcript.append(df.transcription[i])
		transcript_format.append("string")
		transcript_opts.append(None)

	new_df = pd.DataFrame({"ID":ID, "duration":duration, "wav":wav, "wav_format":wav_format, "wav_opts":wav_opts, "spk_id":spk_id, "spk_id_format":spk_id_format,"spk_id_opts":spk_id_opts,"semantics":semantics, "semantics_format":semantics_format,"semantics_opts":semantics_opts, "transcript":transcript, "transcript_format":transcript_format, "transcript_opts":transcript_opts})
	new_filename = folder + split + "-type%d.csv" % type
	new_df.to_csv(new_filename,index=False)
	print(new_filename)

