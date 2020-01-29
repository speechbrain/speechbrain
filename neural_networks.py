# To do:
# - model saving/restarting
# add forward phase+ kaldi decoding and scoring
# test of beluga with different lr strategies
# check performance and speed.
# - refactoring nn (implement linear with conv, implement replicate)
# - test
# - implement, conv1d, conv2d, sincnet, GRU, LSTM, LiGRU, RNN, cuLSTM, cuRNN, QRNN
#   tests
# - Minimal examples: autoencoder, spk-id, ASR.
# - Documentation




"""
 -----------------------------------------------------------------------------
 neural_networks.py

 Description: This library gathers classes that implement neural networks.
              All the classes are designed
              with the same input/output arguments such that they can be
              called within configuration files.
 -----------------------------------------------------------------------------
"""

# Importing libraries
import sys
import torch
import numpy as np
import torch.nn as nn


from utils import (
    check_opts,
    check_inputs,
    logger_write,
)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def act_fun(act_type):

    if act_type == "relu":
        return nn.ReLU()

    if act_type == "tanh":
        return nn.Tanh()

    if act_type == "sigmoid":
        return nn.Sigmoid()

    if act_type == "leaky_relu":
        return nn.LeakyReLU(0.2)

    if act_type == "elu":
        return nn.ELU()

    if act_type == "softmax":
        return nn.LogSoftmax(dim=1)

    if act_type == "linear":
        return nn.LeakyReLU(1)  # initializzed like this, but not used in forward!


class small_MLP(nn.Module):
    
    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        logger=None,
        first_input=None,
    ):
        super(small_MLP, self).__init__()

        # Logger setup
        self.logger = logger

        # Here are summarized the expected options for this class
        
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "device": ("one_of(cpu,cuda)", "optional", "None"),
        }

        # Check, cast , and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )
        

        # Definition of the expected input
        self.expected_inputs = ["torch.Tensor"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )
        
        
        fea_dim=first_input[0].shape[1]
  
        # Parameters
        self.wx = nn.Linear(fea_dim, 1973, bias=True).to(self.device)
        
        self.softmax=nn.LogSoftmax(dim=1)


    def forward(self, input_lst):
        
        # Reading input _list
        x = input_lst[0]
        
        
        
        x=x.transpose(1,2)
        
        x_or=x
        
        x=x.reshape(x.shape[0]*x.shape[1],x.shape[2])
        
        x = self.softmax(self.wx(x))
        
        x=x.reshape(x_or.shape[0],x_or.shape[1],x.shape[1])


        
        return [x]
        
class MLP(nn.Module):
    
    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        logger=None,
        first_input=None,
    ):
        super(MLP, self).__init__()

        # Logger setup
        self.logger = logger

        # Here are summarized the expected options for this class
        
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "device": ("one_of(cpu,cuda)", "optional", "None"),
            "dnn_lay": ("int_list(1,inf)", "mandatory"),
            "dnn_drop": ("float_list(0.0,1.0)", "mandatory"),
            "dnn_use_batchnorm": ("bool_list", "mandatory"),
            "dnn_use_batchnorm_inp": ("bool", "mandatory"),
            "dnn_use_laynorm": ("bool_list", "mandatory"),
            "dnn_use_laynorm_inp": ("bool", "mandatory"),  
            "dnn_act": ("str_list", "mandatory"),  
        }

        # Check, cast , and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )
        

        # Definition of the expected input
        self.expected_inputs = ["torch.Tensor","str"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )
        
        self.N_batches=first_input[0].shape[0]
        self.input_dim=first_input[0].shape[-2]
        
        # additional variables
        self.wx = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        self.ln = nn.ModuleList([])
        self.act = nn.ModuleList([])
        self.drop = nn.ModuleList([])
        
        # initialize layer normalization
        if self.dnn_use_laynorm_inp:
            self.ln0 = LayerNorm(self.input_dim).to(self.device)
        
        
        # input batch normalization
        if self.dnn_use_batchnorm_inp:
            self.bn0 = nn.BatchNorm1d(self.input_dim, momentum=0.05).to(self.device)

        self.N_dnn_lay = len(self.dnn_lay)

        current_input = self.input_dim

        # Initialization of hidden layers

        for i in range(self.N_dnn_lay):

            # dropout
            self.drop.append(nn.Dropout(p=self.dnn_drop[i]).to(self.device))

            # activation
            self.act.append(act_fun(self.dnn_act[i]))

            add_bias = True

            # layer norm initialization
            self.ln.append(LayerNorm(self.dnn_lay[i]).to(self.device))
            self.bn.append(nn.BatchNorm1d(self.dnn_lay[i], momentum=0.05).to(self.device))

            if self.dnn_use_laynorm[i] or self.dnn_use_batchnorm[i]:
                add_bias = False

            # Linear operations
            self.wx.append(nn.Linear(current_input, self.dnn_lay[i], bias=add_bias).to(self.device))

            # weight initialization
            self.wx[i].weight = torch.nn.Parameter(
                torch.Tensor(self.dnn_lay[i], current_input).uniform_(
                    -np.sqrt(0.01 / (current_input + self.dnn_lay[i])),
                    np.sqrt(0.01 / (current_input + self.dnn_lay[i])),
                ).to(self.device)
            )
            self.wx[i].bias = torch.nn.Parameter(torch.zeros(self.dnn_lay[i]).to(self.device))

            current_input = self.dnn_lay[i]

        self.out_dim = current_input


    def forward(self, input_lst):
        
        # Reading input _list
        x = input_lst[0]
        
        mode = input_lst[1]
        
        if mode=='train':
            self.train()
        
        if mode=='valid':
            self.eval()
                    
        x=x.transpose(1,2)
        
        x_or=x
        
        x=x.reshape(x.shape[0]*x.shape[1],x.shape[2])
        
                
        # Adding signal to gpu or cpu
        x = x.to(self.device)

        # Applying Layer/Batch Norm
        if bool(self.dnn_use_laynorm_inp):
            x = self.ln0((x))

        if bool(self.dnn_use_batchnorm_inp):

            x = self.bn0((x))

        for i in range(self.N_dnn_lay):

            if self.dnn_use_laynorm[i] and not (self.dnn_use_batchnorm[i]):
                x = self.drop[i](self.act[i](self.ln[i](self.wx[i](x))))

            if self.dnn_use_batchnorm[i] and not (self.dnn_use_laynorm[i]):
                x = self.drop[i](self.act[i](self.bn[i](self.wx[i](x))))

            if self.dnn_use_batchnorm[i] == True and self.dnn_use_laynorm[i] == True:
                x = self.drop[i](self.act[i](self.bn[i](self.ln[i](self.wx[i](x)))))

            if self.dnn_use_batchnorm[i] == False and self.dnn_use_laynorm[i] == False:
                x = self.drop[i](self.act[i](self.wx[i](x)))

        
        x=x.reshape(x_or.shape[0],x_or.shape[1],x.shape[1])
        
        return [x]
    
    
class compute_cost(nn.Module):
    
    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        logger=None,
        first_input=None,
    ):
        super(compute_cost, self).__init__()

        # Logger setup
        self.logger = logger

        # Here are summarized the expected options for this class
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "cost_type": ("one_of(nll,mse)", "mandatory"),
            "device": ("one_of(cpu,cuda)", "optional", "None"),
        }

        # Check, cast , and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )
        
        # Definition of the expected input
        self.expected_inputs = ["torch.Tensor",'torch.Tensor']

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )
        
        if self.cost_type=='nll':
            self.cost=torch.nn.NLLLoss()
        
        
    def forward(self, input_lst):
        
        # Reading input _list
        prob = input_lst[0]
        
        lab = input_lst[1]
        
        prob=prob[:,0:lab.shape[1],:]
        
        
        # Adding signal to gpu or cpu
        prob = prob.to(self.device)
        lab = lab.to(self.device).long()
        
        # reshaping
        lab=lab.reshape(lab.shape[0]*lab.shape[1])
        prob=prob.reshape(prob.shape[0]*prob.shape[1],prob.shape[2])
        
        
        loss = self.cost(prob,lab)
                
        return [loss]
    
    
    
class optimize(nn.Module):
    
    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        logger=None,
        first_input=None,
    ):
        super(optimize, self).__init__()

        # Logger setup
        self.logger = logger

        # Here are summarized the expected options for this class
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "optimizer_type": ("one_of(rmsprop)", "mandatory"),
            "learning_rate": ("float", "mandatory"),
            "device": ("one_of(cpu,cuda)", "optional", "None"),
            "alpha": ("float", "optional", "0.95"),
            "eps": ("float", "optional", "1e-8"),
            "weight_decay": ("int", "optional", "0"),
            "momentum": ("float", "optional", "0.0"),
            "centered": ("bool", "optional", "False"),
        }

        # Check, cast , and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )
        
        # Analysis of the first input 
        if len(first_input)==0:
            err_msg = (
                'The class optimize expected in input a list of neural '
                'classes (nn.Module). Got an empty list.'
            )

            logger_write(err_msg, logfile=logger)
        
        # Making sure the input is class with parameters to optimize
        param_lst=[]
        
        for inp in first_input:
            try:
                param_lst=param_lst+list(inp.parameters())
            except Exception:
                err_msg = (
                        'The class optimize expected in input a list of ',
                        'neural classes (nn.Module), but %s has no parameters'
                ) % (inp)

                logger_write(err_msg, logfile=logger)
                
        
        # Initialization of the optimizer
        if self.optimizer_type=='rmsprop':
            
            self.optim=torch.optim.RMSprop(param_lst, 
                                      lr=self.learning_rate, 
                                      alpha=self.alpha, 
                                      eps=self.eps, 
                                      weight_decay=self.weight_decay, 
                                      momentum=self.momentum, 
                                      centered=self.centered)
            
        
    def forward(self, input_lst):
        
        self.optim.step()
        
        self.optim.zero_grad()
                
        return []
    

class lr_annealing(nn.Module):
    
    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        logger=None,
        first_input=None,
    ):
        super(lr_annealing, self).__init__()

        # Logger setup
        self.logger = logger

        # Here are summarized the expected options for this class
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "annealing_type": ("one_of(constant,poly,newbob,custom)", 
                               "mandatory"),
            "annealing_factor": ("float", "optional","0.50"),
            "improvement_threshold": ("float", "optional","0.0025"),
            "lr_at_epoch": ("float_list", "optional","None"),
            "N_epochs": ("int", "optional","None"),
            "decay": ("float", "optional","None"),
        }

        # Check, cast , and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )
        
        # Definition of the expected input
        self.expected_inputs = ["int","torch.Tensor","class"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )
        
        # Additional checks on the input when annealing_type=='custom
        if self.annealing_type=='custom':
            
            # Making sure that lr_at_epoch is a list that contains a number
            # of elements equal to the number of epochs.
            if self.lr_at_epoch is None:
                
                err_msg = (
                    'The field lr_at_epoch must be a list composed of'
                    'N_epochs elements when annealing_type=custom'
                )
    
                logger_write(err_msg, logfile=logger)
            
            if self.N_epochs is None:
                
                err_msg = (
                    'The field N_epochs must be specified when'
                    'annealing_type=custom=custom'
                )
    
                logger_write(err_msg, logfile=logger)
                
            if len(self.lr_at_epoch)!=self.N_epochs:
                
                err_msg = (
                    'The field lr_at_epoch must be a list composed of'
                    'N_epochs elements when annealing_type=custom.'
                    'Got a list of %i elements (%i expected)'
                ) %(len(self.lr_at_epoch),len(self.N_epochs))
    
                logger_write(err_msg, logfile=logger)
 
        # Additional checks on the input when annealing_type=='poly'
        if self.annealing_type=='poly':
            if self.decay is None and self.N_epochs is None:
                err_msg = (
                    'The field N_epochs must be specified when'
                    'annealing_type=poly and decay is not specified externally'
                )
    
                logger_write(err_msg, logfile=logger)
                
            
        # Initalizing the list that stored the losses/errors
        self.losses=[]


    def forward(self, input_lst):
        
        # Current epoch
        current_epoch=input_lst[0]
        
        # Current loss
        current_loss=input_lst[1]
        
        # Current optimizer
        current_opts=input_lst[2]
        
        for opt in current_opts:
            
            # Current learning rate
            current_lr=opt.param_groups[0]['lr']
            
            # Managing newbob annealing
            if self.annealing_type=='newbob':
                
                if len(self.losses)>0:
                    if (self.losses[-1]-current_loss)/self.losses[-1] < self.improvement_threshold:
                        current_lr=current_lr*self.annealing_factor

            # Managing newbob annealing
            if self.annealing_type=='custom':
                current_lr=self.lr_at_epoch[current_epoch]
                
            if self.annealing_type=='poly':
                
                if self.decay is None:
                    self.decay=self.current_lr/self.N_epochs
                
                current_lr=current_lr/(1.0+self.decay*current_epoch)
                
                      
                                    
            opt.param_groups[0]['lr'] = current_lr
        
        # Appending current loss
        self.losses.append(current_loss)
                
        return []
    

