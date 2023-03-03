This recipe provides a baseline code to do multi-tasked Self-supervised speech representation learning.
A few signal workers are already available, but adding new tasks is easy, and requires a few lines of code only ( apart from the extraction of these workers ) 
To launch a code, you can try directly with simple workers like mel spectrograms reconstruction or MFCCs. Values for other workers should be stocked in csv files as explained in the yaml file. Those csv files are pandas pickle where calling the workers name should lead to their values for the considered speech sample. 
One example of these csv files, extracted on a file from the LibriSpeech dataset, is provided.
