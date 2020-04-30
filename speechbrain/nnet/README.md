## Architectures
The library in *speechbrain.nnet/architectures.py* contains different types of neural networks, that can
be used to implement **fully-connected**, **convolutional**, and **recurrent models**.  All the models are designed to be combined to create more complex neural architectures. 
Please, take a look into the following configuration files to see an example different architectures:

| Architecture   |      Configuration file      | 
|----------|:-------------:|
| MLP |  *cfg/minimal_examples/neural_networks/spk_id/Basic_MLP.cfg* | 
| CNN |    -  |
| SincNet |    -  |
| RNN | - |
| GRU | - |
| Li-GRU | - |
| LSTM | - |
| CNN+MLP+RNN | - |

In the spk_id example introduced in the previous section, one can simply switch from architecture to another. 

## Normalization
SpeechBrain implements different modalities to normalize neural networks in speechbrain/speechbrain/nnet/normalization.py. The function currently supports:
- **batchnorm**: it applies the standard batch normalization by normalizing the mean and std of the input tensor over the batch axis.
- **layernorm**: it applies the standard layer normalization by normalizing the mean and std of the input tensor over the neuron axis.
- **groupnorm"**: it applies group normalization over a mini-batch of inputs. See torch.nn documentation for more info.

- **instancenorm**: it applies instance norm over a mini-batch of inputs. It is similar to layernorm, but different statistics for each channel are computed.

- **localresponsenorm**: it applies local response normalization over an input signal composed of several input planes. See torch.nn documentation for more info.

An example of an MLP coupled with batch normalization can be found here:

## Losses
The loss is a measure that estimates "how far" are the target labels from the current network output.
The loss (also known as *cost function* or *objective*) are scalars from which the gradient is computed. 

In Speechbrain the cost functions are implemented in *speechbrain/speechbrain/nnet/losses.py*. Currently, the following losses are supported:
- **nll**: it is the standard negative log-likelihood cost (categorical cross-entropy).
- **mse**: it is the mean squared error between the prediction and the target.
- **l1**:  it is the l1 distance between the prediction and the target.
- **ctc**:  it is the CTC function used for sequence-to-sequence learning. It sums p over all the possible alignments between targets and predictions.
- **error**:  it is the standard classification error.

Depending on the typology of the problem to address, a different loss function must be used. For instance, regression problems typically use *l1* or *mse* losses, while classification problems require often *nll*. 

An important option is *avoid_pad*. when True, the time steps corresponding to zero-padding are not included in the cost function computation.

## Optimizers
In SpeechBrain the optimizers are implemented in *speechbrain/speechbrain/nnet/optimizers.py*. All the basic optimizers such as *Adam*, *SGD*, *Rmseprop* (along with their most popular variations) are implemented. See the function documentation for a complete list of all the available alternatives. 

 As you can see, the optimizer takes in input the names of the functions contaning the parameters.
 After updating the parameters, the optimizer automatically calls the function *zero_grad()* to add zeros in the gradient buffers (and avoid its accumulation in the next batch of data).

## Learning rate scheduler
During neural network training, it is very convenient to change the values of the learning rate. Typically, the learning rate is annealed during the training phase. In *speechbrain/speechbrain/nnet/lr_scheduling.py* we implemented some learning rate schedules.  In particular, the following strategies are currently available:

- **newbob**: the learning rate is annealed based on the validation performance. In particular:
    ```
    if (past_loss-current_loss)/past_loss < impr_threshold:
    lr=lr * annealing_factor
    ```                                              
                                              
- **time_decay**: linear decay over the epochs:
    ```
    lr=lr_ini/(epoch_decay*(epoch)) 
    ```

- **step_decay**:  decay over the epochs with the selected epoch_decay factor
     ```
    lr=self.lr_int*epoch_decay/((1+epoch)/self.epoch_drop)
     ```

- **exp_decay**:   exponential decay over the epochs selected epoch_decay factor
    ```
    lr=lr_ini*exp^(-self.exp_decay*epoch)
    ```
- **custom**:   the learning rate is set by the user with an external array (with length equal to the number of epochs)
