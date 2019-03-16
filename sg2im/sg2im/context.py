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
    Scene-graph context network that pools features generated
    GCN and feeds through fc layer to generate embedding
    """
    def __init__(self, 
                 input_dim=128, 
                 output_dim=8):
        super(Context, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, vecs):
        """
        Inputs:
        - vecs: Tensor of shape (O, D) giving vectors
        
        Returns:
        - out: Tensor of shape (D,)
        """
        O, D = vecs.size()
        pooled = vecs.sum(0)
        out = self.fc(pooled)
        return out
        
        
        

