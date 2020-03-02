"""
 -----------------------------------------------------------------------------
 optimizers.py

 Description: This library implements different optimizers.
 -----------------------------------------------------------------------------
"""

import torch
import torch.nn as nn
from ..data_io.data_io import recovery
from ..utils import check_opts, logger_write


class optimize(nn.Module):
    """
     -------------------------------------------------------------------------
     nnet.optimizers.optimize (author: Mirco Ravanelli)

     Description:  This function implements different optimizers.
                   It supports standard optimizers such as adam, sgd, rmseprop,
                   and some of their variations such as as adamw, adamax, 
                   adadelta. The function takes in input some neural networks
                   and updates their parameters according to the optimization
                   algorithm adopted.
                 
     Input (init):  - config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.

                           - optimizer_type (one_of(rmsprop,adam,adamw,adamax,
                                             adadelta,sgd,rprop), mandatory):
                               it is the type of optimizer to be used. 
                               Refer to torch.nn documentation of a more
                               detailed description of each optimizer.


                           - learning_rate (float, mandatory):
                               it is the learning rate used to update the
                               parameters.
                               
                           - alpha (float, optional, Default:0.95):
                               it is used the smoothing constant used in
                               rmseprop. 
                               
                           - betas (float_list, optional, Default:0.95):
                                are coefficients used for computing running 
                                averages of gradient and its square in adam
                                optimizer and its variations.
                                
                           - etas (float_list, optional, Default:0.5,1.2):
                               yt is used in Rprop optimizer. It is a
                               pair of (etaminus, etaplis), that are 
                               multiplicative increase and decrease factors.
                               
                           - eps (float, optional, Default:1e-8):
                               it is the numerical stability factor.
                               
                           - step_sizes (float_list, optional, 
                                         Default: 1e-06, 50):
                              It is used in rprop optimizer and contains a 
                              pair of minimal and maximal allowed step sizes.
                              
                           - weight_decay (int, optional, Default: 0):
                               it is the weight decay (L2 penalty) factor
                               used as as additionaly loss.
                               
                           - momentum (float, optional, Default: 0.0):
                              it is the momentum factor for the optimizers.
                              
                           - dampening (float, optional, Default: 0.0):
                               it is  dampening facror for SGD momentum.
                               
                           - rho (float, optional, Default: 0.0):
                               it is used in adadelta and it is the coefficient 
                               used for computing a running average of 
                               squared gradients.
                               
                           - centered (bool, optional, Default: False): 
                               if True, compute the centered RMSProp, the 
                               gradient is normalized by an estimation of 
                               its variance.
                               
                           - amsgrad (bool, optional, Default: False):
                                if True it uses the AMSGrad variant of the
                                adam optimizer.
                                
                           - nesterov (bool, optional, Default: False): 
                                it enables Nesterov momentum for SGD.

                           - recovery (type: bool, optional, Default:True):
                               if True, the system restarts from the last 
                               epoch correctly executed.                               
                                
                               
                   - funct_name (type, str, optional, default: None):
                       it is a string containing the name of the parent
                       function that has called this method.

                   - global_config (type, dict, optional, default: None):
                       it a dictionary containing the global variables of the
                       parent config file.

                   - logger (type, logger, optional, default: None):
                       it the logger used to write debug and error messages.
                       If logger=None and root_cfg=True, the file is created
                       from scratch.

                   - first_input (type, list, optional, default: None)
                      this variable allows users to analyze the first input
                      given when calling the class for the first time. In this
                      case, it contains the list of neural model to optimize.


     Input (call): - inp_lst(type, list, mandatory):
                       it is a list containing the neural networks to optimize.
                       


     Output (call): - None
                       This function returns "None" when called. It directly
                       changes the parameters of the neural networks listed
                       in the input.
                       
                       
     Example:   import torch
                from lib.nnet.architectures import linear
                from lib.nnet.architectures import activation
                from lib.nnet.losses import compute_cost
                from lib.nnet.optimizers import optimize
                
                # Definition the input tensor
                inp_tensor = torch.rand([1,660,3])
                
                # Initialization of the linear class
                config={'class_name':'lib.nnet.architectures.linear',
                        'n_neurons':'4'}

                model=linear(config,first_input=[inp_tensor])
                
                
                # Initialization of the log_softmax class            
                config={'class_name':'lib.nnet.architectures.activation',
                        'act_type':'log_softmax',
                        }

                softmax=activation(config, first_input=[inp_tensor])
                
                
                # Initialization of the loss function
                config={'class_name':'lib.nnet.losses.compute_cost',
                        'cost_type':'nll'}
                
                cost=compute_cost(config)
                
                # Initialization of the optimizer
                config={'class_name':'lib.nnet.optimizers.optimizer',
                        'optimizer_type': 'sgd',
                        'learning_rate': '0.01'
                        }

                optim=optimize(config, first_input=[model])              
                
                
                # Computatitions of the prediction for the current input
                pre_act=model([inp_tensor])
                pred = softmax([pre_act])
                
                # fake label
                label=torch.FloatTensor([0,1,3]).unsqueeze(0)
                lengths=torch.Tensor([1.0])
                
                out_cost= cost([pred,label,lengths])
                
                print(out_cost)
                
                # back propagation
                out_cost.backward()
                
                print(list(model.parameters()))
                
                # applying optimization
                optim([model])
                
                print(list(model.parameters()))
                
     """    
    def __init__(
        self,
        config,
        funct_name=None,
        global_config=None,
        functions=None,
        logger=None,
        first_input=None,
    ):
        super(optimize, self).__init__()

        # Logger setup
        self.logger = logger
        
        if global_config is not None:
            self.output_folder = global_config['output_folder']
        
        self.funct_name = funct_name

        # Here are summarized the expected options for this class
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "optimizer_type": ("one_of(rmsprop,adam,adamw,adamax,adadelta,sgd,rprop)", "mandatory"),
            "learning_rate": ("float", "mandatory"),
            "recovery": ("bool", "optional","True"),
            "alpha": ("float", "optional", "0.95"),
            "betas": ("float_list", "optional", "0.9,0.999"),
            "etas": ("float_list", "optional", "0.5,1.2"),
            "step_sizes": ("float_list", "optional", "1e-06, 50"),
            "eps": ("float", "optional", "1e-8"),
            "weight_decay": ("int", "optional", "0"),
            "momentum": ("float", "optional", "0.0"),
            "dampening": ("float", "optional", "0.0"),
            "rho": ("float", "optional", "0.0"),
            "initial_accumulator_value": ("float", "optional", "0.0"),
            "centered": ("bool", "optional", "False"),
            "amsgrad": ("bool", "optional", "False"),
            "nesterov": ("bool", "optional", "False"),

            
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
        
        # Storing all the parameters to updated in the param_lst
        for inp in first_input:
            
            try:
                param_lst=param_lst+list(inp.parameters())
            except Exception:
                
                err_msg = (
                    'The class optimize expected in input a list of'
                    'neural classes (nn.Module), but %s has no parameters'
                    % (inp)
                )

                logger_write(err_msg, logfile=logger)
                
        
        # Initialization of the rmsprop optimizer
        if self.optimizer_type=='rmsprop':
            
            self.optim=torch.optim.RMSprop(param_lst, 
                                      lr=self.learning_rate, 
                                      alpha=self.alpha, 
                                      eps=self.eps, 
                                      weight_decay=self.weight_decay, 
                                      momentum=self.momentum, 
                                      centered=self.centered)
           
        # Initialization of the adam optimizer    
        if self.optimizer_type=='adam':
            
            self.optim=torch.optim.Adam(param_lst, 
                                      lr=self.learning_rate, 
                                      betas=tuple(self.betas), 
                                      eps=self.eps, 
                                      weight_decay=self.weight_decay, 
                                      amsgrad=self.amsgrad)
            
        # Initialization of the adamw optimizer
        if self.optimizer_type=='adamw':
            
            self.optim=torch.optim.AdamW(param_lst, 
                                      lr=self.learning_rate, 
                                      betas=tuple(self.betas), 
                                      eps=self.eps, 
                                      weight_decay=self.weight_decay, 
                                      amsgrad=self.amsgrad)

        # Initialization of the adamax optimizer
        if self.optimizer_type=='adamax':
            
            self.optim=torch.optim.Adamax(param_lst, 
                                      lr=self.learning_rate, 
                                      betas=tuple(self.betas), 
                                      eps=self.eps) 

        # Initialization of the adadelta optimizer
        if self.optimizer_type=='adadelta':
            
            self.optim=torch.optim.Adadelta(param_lst, 
                                      lr=self.learning_rate, 
                                      rho = self.rho,
                                      eps=self.eps,
                                      weight_decay=self.weight_decay)
            
        # Initialization of the sgd optimizer
        if self.optimizer_type=='sgd':
            
            self.optim=torch.optim.SGD(param_lst, 
                                      lr=self.learning_rate, 
                                      momentum=self.momentum, 
                                      dampening=self.dampening, 
                                      weight_decay=self.weight_decay, 
                                      nesterov=self.nesterov)

        # Initialization of the rprop optimizer
        if self.optimizer_type=='rprop':
            
            self.optim=torch.optim.Rprop(param_lst, 
                                      lr=self.learning_rate, 
                                      etas=tuple(self.etas), 
                                      step_sizes=tuple(self.step_sizes))
            
        # Automatic recovery
        if global_config is not None:
            recovery(self)
        
        
    def forward(self, input_lst):
             
        # Gradient combination for the multi-gpu case           
        self.sum_grad_multi_gpu(input_lst)
        
        # Parameter update
        self.optim.step()
        
        # Zeroing gradient buffers
        self.optim.zero_grad()
                
        return
    
    
    def sum_grad_multi_gpu(self,input_lst):
         """
         ----------------------------------------------------------------------
         nnet.optimizers.optimize.sum_grad_multi_gpu (author: Mirco Ravanelli)
    
         Description: This support function is used in the multi-gpu scenario
                      and sums all the gradients coming from the different gpus.
    
         Input (call):    - input_lst (type: list, mandatory):
                               list of all the neural models to optimize.
    
    
         Output (call):  None:
                          the gradient is directly updated in the reference 
                          device (which is by default cuda:0).
                           
         ----------------------------------------------------------------------
         """ 
         
         # Loops over all the input models
         for inp in input_lst:
             
            # Check if the computations are multi-gpu 
            if hasattr(inp, 'multi_gpu_models'):
                
                # list of all the parameters
                for index, param in enumerate(inp.parameters()):
                    
                    first=True
                    
                    # look for the models replicated over the various gpus
                    for model in inp.multi_gpu_models:
                        
                        # model parameter in the current gpu
                        par_gpu= list(model.parameters())[index].grad
                        
                        if first:
                            par_sum=par_gpu.to('cuda:0')
                            first=False
                        else:
                            par_sum=par_sum+par_gpu.to('cuda:0')
                    
                    # Summing up all the gradients
                    param.grad=param.grad +  par_sum
    
