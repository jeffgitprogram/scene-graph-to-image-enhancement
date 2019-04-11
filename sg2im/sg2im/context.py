import torch
import torch.nn as nn

"""
Module for scene graph context

!!! 
I think the scene-graph context paper only pools predicate embeddings
from GCN because if you look at the code for sg-to-img (model.py line 140)
they never use pred_vecs
!!!

"""

class Context(nn.Module):
    """
    Context network that pools hidden units generated from
    LSTM and feeds through fc layer to generate embeddings
    """
    def __init__(self, 
                 input_dim=1500,
                 noise_dim=0,
                 output_dim=128,
                 H=64,
                 W=64):
        super(Context, self).__init__()
        self.input_dim = input_dim
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.H = H
        self.W = W
        self.fc = nn.Linear(input_dim + noise_dim, output_dim * H * W)
        self.leaky_relu = nn.RReLU()
    
    def forward(self, lstm_hidden):
        """
        Inputs:
        - lstm_hidden: Tensor of shape (N, self.input_dim)
        
        Returns:
        - output: Tensor of shape (N, self.output_dim, H, W)
        """
        N, _ = lstm_hidden.size()
        if noise_dim > 0:
            noise_shape = (N, self.noise_dim)
            noise = torch.randn(noise_shape)
            input = torch.cat([lstm_hidden, noise], dim=1)
        output = self.leaky_relu(self.fc(lstm_hidden))
        return output.reshape(N, self.output_dim, self.H, self.W)
        
        
        

