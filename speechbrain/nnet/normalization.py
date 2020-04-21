"""
Neural normalization strategies.
"""

import torch
import torch.nn as nn


class normalize(nn.Module):
    """
     -------------------------------------------------------------------------
     nnet.normalization.normalize (author: Mirco Ravanelli)

     Description:  This function implements different normalization techniques
                   such as batchnorm, layernorm, groupnorm, instancenorm, and
                   localresponsenorm.

     Input (init):  - config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.

                           - norm_type (one_of(batchnorm,layernorm,groupnorm,
                                        instancenorm,localresponsenorm,
                                        mandatory):
                               it is the type of normalization used.

                               "batchnorm": it applies the standard batch
                                            normalization by normalizing
                                            mean and std of the input tensor
                                            over the  batch axis.

                               "layernorm": it applies the standard layer
                                            normalization by normalizing
                                            mean and std of the input tensor
                                            over the neuron axis.

                               "groupnorm": it applies group normalization
                                            over a mini-batch of inputs.
                                            See torch.nn documentation for
                                            more info.

                               "instancenorm": it applies instance norm
                                            over a mini-batch of inputs.
                                            It is similar to layernorm,
                                            but different statistic for each
                                            channel are computed.

                               "localresponsenorm": it applies local response
                                            normalization over an input signal
                                            composed of several input planes.
                                            See torch.nn documentation for
                                            more info.

                           - eps (float(0,inf), optional, Default: 1e-05):
                               this value is added to std deviation estimation
                               to improve the numerical stability.

                           - momentum (float(0,inf), optional, Default: 0.1):
                               It is a value used for the running_mean and
                               running_var computation.

                           - alpha (float(0,inf), optional, Default: 0.0001"):
                               alpha factor for localresponsenorm.

                           - beta (float(0,inf), optional, Default: 0.75"):
                               beta factor for localresponsenorm.

                           - k (float(0,inf), optional, Default: 1.0"):
                               k factor for localresponsenorm.

                           - neigh_ch (int(1,inf), optional, Default: 2"):
                                it is amount of neighbouring channels used for
                                localresponse normalization.

                           - affine (bool, optional, Default: "True"):.
                             when set to True, the affine parameters are
                             learned.

                           - elementwise_affine (bool, optional, \
                               Default: "True"):
                             it is used for the layer normalization. If True,
                             this module has learnable per-element affine
                             parameters initialized to ones (for weights)
                             and zeros (for biases).

                           - track_running_stats (bool, optional, \
                               Default: "True"):
                               when set to True, this module tracks the
                               running mean and variance, and when set to
                               False, this module does not track
                               such statistics. Set it to True for batch
                               normalization and to False for instancenorm.

                           - num_groups (int(1,inf), optional, Default: 1"):
                                it is number of groups to separate the
                                channels into for the group normalization.

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
                      given when calling the class for the first time.


     Input (call): - inp_lst(type, list, mandatory):
                       it is a list containing the tensor x to normalize.


     Output (call): - x_n (type: torch.Tensor)
                       the function returns the normalized tensor.



     Example:   import torch
                from speechbrain.nnet.normalization import normalize

                # input tensor
                inp_tensor = torch.rand([4,660,3])

                # normalization initialization
                config={'class_name':'speechbrain.nnet.normalization.normalization',
                        'norm_type':'batchnorm',
                        'track_running_stats': 'False'}

                # Initialization of the linear class
                batchnorm=normalize(config, first_input=[inp_tensor])

                print(inp_tensor[:,0,0].mean())
                print(inp_tensor[:,0,0].std())
                out =  batchnorm([inp_tensor]

                print(out[:,0,0].mean())
                print(out[:,0,0].std())

     """

    def __init__(
        self,
        norm_type,
        eps=1e-05,
        momentum=0.1,
        alpha=0.0001,
        beta=0.75,
        k=1.0,
        affine=True,
        elementwise_affine=True,
        track_running_stats=True,
        num_groups=1,
        neigh_ch=2,
        output_folder=None,
    ):
        super().__init__()

        self.norm_type = norm_type
        self.eps = eps
        self.momentum = momentum
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.affine = affine
        self.elementwise_affine = elementwise_affine
        self.track_running_stats = track_running_stats
        self.num_groups = num_groups
        self.neigh_ch = neigh_ch
        self.output_folder = output_folder

        # Reshaping when input to batchnorm1d is 3d makes it faster
        self.reshape = False

    def init_params(self, first_input):

        # Initializing bachnorm
        if self.norm_type == "batchnorm":
            self.norm = self.batchnorm(first_input)

        # Initializing groupnorm
        if self.norm_type == "groupnorm":
            n_ch = first_input.shape[1]
            self.norm = torch.nn.GroupNorm(
                self.num_groups, n_ch, eps=self.eps, affine=self.affine
            )

        # Initializing instancenorm
        if self.norm_type == "instancenorm":
            self.norm = self.instancenorm(first_input)

        # Initializing layernorm
        if self.norm_type == "layernorm":
            self.norm = torch.nn.LayerNorm(
                first_input.size()[1:-1],
                eps=self.eps,
                elementwise_affine=self.elementwise_affine,
            )

            self.reshape = True

        # Initializing localresponsenorm
        if self.norm_type == "localresponsenorm":
            self.norm = torch.nn.LocalResponseNorm(
                self.neigh_ch, alpha=self.alpha, beta=self.beta, k=self.k
            )

    def forward(self, x):

        # Reshaping (if needed)
        if self.reshape:

            x = x.transpose(1, -1)
            dims = x.shape

            x = x.reshape(dims[0] * dims[1], dims[2])

        # Applying batch normalization
        x_n = self.norm(x)

        # Getting the original dimensionality
        if self.reshape:

            x_n = x_n.reshape(dims[0], dims[1], dims[2])

            x_n = x_n.transpose(1, -1)

        return x_n

    def batchnorm(self, first_input):
        """
         ----------------------------------------------------------------------
         nnet.normalization.normalize.batchnorm (author: Mirco Ravanelli)

         Description:  This support function intializes 1D, 2D, or 2D
                       batch normalization.


         Input (call):
                        - first_input (type:list, mandatory):
                            it is the list containing the first torch.Tensor
                            to normalize.



         Output (call): - norm(type, object):
                           it is the batch normalization object just created.


         Example:   import torch
                    from speechbrain.nnet.normalization import normalize

                    # input tensor
                    inp_tensor = torch.rand([4,660,3])

                    # normalization initialization
                    config={'class_name':'speechbrain.nnet.normalization.\
                        normalization',
                            'norm_type':'batchnorm',
                            'track_running_stats': 'False'}

                    # Initialization of the linear class
                    batchnorm=normalize(config, first_input=[inp_tensor])

                    print(batchnorm.batchnorm)

         """

        # Getting the feature dimension
        fea_dim = first_input.shape[1]

        # Based on the shape of the input tensor I can use 1D,2D, or 3D batchn
        if len(first_input.shape) <= 3:

            # Managing 1D batchnorm
            norm = nn.BatchNorm1d(
                fea_dim,
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )

            if len(first_input.shape) == 3:
                self.reshape = True

        if len(first_input.shape) == 4:

            # Managing 2D batchnorm
            norm = nn.BatchNorm2d(
                fea_dim,
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )

        if len(first_input.shape) == 5:

            # Managing 3D batchnorm
            norm = nn.BatchNorm3d(
                fea_dim,
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )

        return norm.to(first_input.device)

    def instancenorm(self, first_input):
        """
         ----------------------------------------------------------------------
         nnet.normalization.normalize.instancenorm (author: Mirco Ravanelli)

         Description:  This support function intializaes 1D, 2D, or 2D
                       instance normalization.


         Input (call):
                        - first_input (type:list, mandatory):
                            it is the list containing the first torch.Tensor
                            to normalize.



         Output (call): - norm(type, object):
                           it is the instance normalization object just
                           created.


         Example:   import torch
                    from speechbrain.nnet.normalization import normalize

                    # input tensor
                    inp_tensor = torch.rand([4,660,3])

                    # normalization initialization
                    config={'class_name':'speechbrain.nnet.normalization.\
                        normalization',
                            'norm_type':'instancenorm',
                            'track_running_stats': 'False'}

                    # Initialization of the linear class
                    instancenorm=normalize(config, first_input=[inp_tensor])

                    print(instancenorm.instancenorm)

         """
        # Getting the feature dimension
        fea_dim = first_input.shape[1]

        # Based on the shape of the input tensor I can use 1D,2D, or 3D
        # instance normalization

        if len(first_input.shape) == 3:
            # 1D case
            norm = nn.InstanceNorm1d(
                fea_dim,
                eps=self.eps,
                momentum=self.momentum,
                track_running_stats=self.track_running_stats,
            )

        if len(first_input.shape) == 4:
            # 2D case
            norm = nn.InstanceNorm2d(
                fea_dim,
                eps=self.eps,
                momentum=self.momentum,
                affine=self.affine,
                track_running_stats=self.track_running_stats,
            )

        if len(first_input.shape) == 5:
            # 3D case
            norm = nn.InstanceNorm3d(
                fea_dim,
                eps=self.eps,
                momentum=self.momentum,
                track_running_stats=self.track_running_stats,
            )

        return norm


class normalize_posteriors(nn.Module):
    """
     -------------------------------------------------------------------------
     nnet.normalization.normalize_posteriors (author: Mirco Ravanelli)

     Description:  This function normalizes the posterior probabilities
                   using the counts of the given input label. This operation
                   can be useful when likelihood are need instead of
                   posterior probabilities (e.g, when feeding an HMM-DNN
                   decoder).



     Input (init):  - config (type, dict, mandatory):
                       it is a dictionary containing the keys described below.

                           - count_lab (type: str, mandatory):
                               it is the label used to normalize the
                               posterior probabilities. In particular,
                               for each label we pre-compute some counts
                               in global_config['lab_dict']['count_lab'] and
                               we use these counts to re-scale the posterior
                               probabilities with the prior one.


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
                      given when calling the class for the first time.


     Input (call): - inp_lst(type, list, mandatory):
                       it is a list containing the tensor x to normalize.


     Output (call): - pout_norm (type: torch.Tensor)
                       the function returns the normalized probabilities.



     Example:   import torch
                from speechbrain.nnet.normalization import normalize_posteriors

                # input tensor
                pout = torch.rand([4,3,10])

                global_config={}
                global_config['label_dict']={}
                global_config['label_dict']['phn']={}
                global_config['label_dict']['phn']['counts']={}
                global_config['label_dict']['phn']['counts'][0]=1
                global_config['label_dict']['phn']['counts'][1]=2
                global_config['label_dict']['phn']['counts'][2]=2


                # normalization initialization
                config={'class_name':'speechbrain.nnet.normalization.\
                    normalize_posteriors',
                        'count_lab':'phn'}

                # Initialization of the linear class
                norm=normalize_posteriors(config, global_config=global_config)

                print(norm.counts)

                out_n = norm([pout])
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
        super(normalize_posteriors, self).__init__()

        # Setting logger and exec_config
        self.logger = logger

        # Definition of the expected options
        self.expected_options = {
            "class_name": ("str", "mandatory"),
            "count_lab": ("str", "mandatory"),
        }

        # Check, cast , and expand the options
        self.conf = check_opts(
            self, self.expected_options, config, self.logger
        )

        # Expected inputs when calling the class
        self.expected_inputs = ["torch.Tensor"]

        # Check the first input
        check_inputs(
            self.conf, self.expected_inputs, first_input, logger=self.logger
        )

        # load the count dictionary
        if self.count_lab in global_config["label_dict"]:
            self.count_dict = global_config["label_dict"][self.count_lab][
                "counts"
            ]

        else:
            err_msg = (
                "The label specified in the field self.count_lab does not "
                'exists in the label dictionary (global_config["label_dict"])'
            )

            raise ValueError(err_msg)

        # converting dictionary to list
        count_lst = []

        for key in sorted(self.count_dict.keys()):
            count_lst.append(self.count_dict[key])

        # converting list to tensor
        self.counts = torch.Tensor(count_lst)

        # Converting to log counts (we normalize log posterios)
        self.counts = torch.log(self.counts / torch.sum(self.counts))

    def forward(self, input_lst):

        # Reading input_list
        pout = input_lst[0]

        # Transposing the pout tensor
        pout_norm = pout.transpose(1, 2)

        # Moving the counts to the right device
        self.counts = self.counts.to(pout.device)

        # Normalization (we assume log probabilities)
        pout_norm = pout_norm - self.counts

        # Getting the original shape format
        pout_norm = pout_norm.transpose(1, 2)

        return pout_norm
