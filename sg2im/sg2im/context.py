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
        self.relu = nn.ReLu()
    
    def forward(self, vecs, pred_to_img):
        """
        Inputs:
        - vecs: Tensor of shape (O, D) giving vectors
        
        Returns:
        - out: Tensor of shape (D,)
        """
        O, D = vecs.size()
        N = pred_to_img.data.max().item() + 1
        out = torch.zeros(N, D, dtype=vecs.dtype, device=vecs.device)
        idx = pred_to_img.view(O,1).expand(O,D)
        out = out.scatter_add(0, idx, vecs)
        out = self.fc(out)
        # TODO Do we need to add batch-norm?
        out = self.relu(out)
        return out
        
        
        

