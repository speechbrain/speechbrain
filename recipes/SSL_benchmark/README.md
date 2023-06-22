To run a downstream evaluation for a given SSL model on huggingface you will need to  : 
* Change the SSL related values in the run\_benchmark.sh file, specifying the HF hub, the encoder dimension (size of every frame vector), and the number of layers.
* Choose a set of tasks among the ones listed in list\_tasks\_downstreams.txt and for every task a downstream architecture among the existing ones. 
* Change the variable defined in run\_benchmark.sh with two lists of equal sized where to every task  in "ConsideredTasks" corresponds in the same index in "Downstreams" the downstream architecture.
* If you want to run two downstream decoders on the same task, just put it twice in the first list with different corresponding decoders below. 





