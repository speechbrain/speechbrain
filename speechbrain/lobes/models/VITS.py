import torch.nn as nn

class PosteriorEncoder(nn.Module):
    def __init__(self):
        super(PosteriorEncoder, self).__init__()
        pass

    def forward(self):
       return
   

class PriorEncoder(nn.Module):
    def __init__(self):
        super(PriorEncoder, self).__init__()
        pass

    def forward(self):
       return

class StochasticDurationPredictor(nn.Module):
    def __init__(self):
        super(StochasticDurationPredictor, self).__init__()
        pass

    def forward(self):
       return

   
class VITS(nn.Module):
    def __init__(self):
        super(VITS, self).__init__()
        self.prior_encoder = PriorEncoder()
        self.posterior_encoder = PosteriorEncoder()
        

    def forward(self, inputs):
       return