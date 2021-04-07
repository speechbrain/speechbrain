# Training set: train-real
# Test set: test-real, test-synth
for i in {1..5}
do
	python train.py hparams/train.yaml --seed=$i --data_folder=/localscratch/timers-and-such/ --number_of_epochs=50 --train_splits=["train-real"] --experiment=train-real-only
done

# Training set: train-real + train-synth
# Test set: test-real, test-synth
for i in {1..5}
do
	python train.py hparams/train.yaml --seed=$i --data_folder=/localscratch/timers-and-such/ --number_of_epochs=2 --train_splits=["train-real", "train-synth"] --experiment=train-real-and-synth
done

# Training set: train-synth
# Test set: test-real, test-synth, all-real
for i in {1..5}
do
	python train.py hparams/train.yaml --seed=$i --data_folder=/localscratch/timers-and-such/ --number_of_epochs=2 --train_splits=["train-synth"] --experiment=train-synth-only --test_on_all_real=True
done
