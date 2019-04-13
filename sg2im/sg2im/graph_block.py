
import torch
import torch.nn as nn
from sg2im.layers import build_mlp

class GraphBlock(nn.Module):
    def __init__(self, input_dim, output_dim=None, hidden_dim=512, pooling='avg', mlp_normalization='none'):
        super(GraphBlock, self).__init__()
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        self.input_to_hidden = nn.Sequential(
            nn.Linear(in_features=3*input_dim, out_features=3*hidden_dim),
            nn.ReLU()
        )
        self.edgeUpdate = EdgeUpdate(3*hidden_dim, output_dim)
        self.nodeUpdate = NodeUpdate(2*hidden_dim+output_dim, output_dim)
        self.hidden_to_out = nn.Sequential(
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
            nn.ReLU()
        )
        
    def forward(self, obj_vecs, pred_vecs, edges):
        dtype, device = obj_vecs.dtype, obj_vecs.device
        O, T = obj_vecs.size(0), pred_vecs.size(0)
        Din, H, Dout = self.input_dim, self.hidden_dim, self.output_dim
        
        # Get hidden representation of triples
        s_idx = edges[:,0].contiguous()
        o_idx = edges[:,1].contiguous()
        
        cur_s_vecs = obj_vecs[s_idx]
        cur_o_vecs = obj_vecs[o_idx]
        # New shape is (T, 3 * H)
        cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
        hidden_t_vecs = self.input_to_hidden(cur_t_vecs)
        
        # Update edge attributes
        hidden_s_vecs = hidden_t_vecs[:, :H]
        hidden_p_vecs = hidden_t_vecs[:, H:2*H] #hidden_t_vecs[:, H:(H+Dout)]
        hidden_o_vecs = hidden_t_vecs[:, 2*H:3*H] #hidden_t_vecs[:, (H+Dout) : (2*H+Dout)]
        # New shape is (T, Dout)
        new_pred_vecs = self.edgeUpdate(hidden_s_vecs, hidden_p_vecs, hidden_o_vecs)
        
        # Aggregate edge attributes per node and update note attributes
        # New shape is (T, 2 * H)
        new_t_vecs = self.nodeUpdate(hidden_s_vecs, new_pred_vecs, hidden_o_vecs)
        new_s_vecs = new_t_vecs[:, :H]
        new_o_vecs = new_t_vecs[:, H:2*H]
        
        pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)
        s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
        o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)
        
        if self.pooling == 'avg':
            # Figure out how many times each object has appeared, again using
            # some scatter_add trickery.
            obj_counts = torch.zeros(O, dtype=dtype, device=device)
            ones = torch.ones(T, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, s_idx, ones)
            obj_counts = obj_counts.scatter_add(0, o_idx, ones)

            # Divide the new object vectors by the number of times they
            # appeared, but first clamp at 1 to avoid dividing by zero;
            # objects that appear in no triples will have output vector 0
            # so this will not affect them.
            obj_counts = obj_counts.clamp(min=1)
            pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)
        
        # Send pooled object vectors through net2 to get output object vectors,
        # of shape (O, Dout)
        new_obj_vecs = self.hidden_to_out(pooled_obj_vecs)
        return new_obj_vecs, new_pred_vecs
        
        
class EdgeUpdate(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EdgeUpdate, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, s_vecs, pred_vecs, o_vecs):
        t_vecs = torch.cat([s_vecs, pred_vecs, o_vecs],dim=1)
        out = self.linear(t_vecs)
        out = self.relu(out)
        return out
    
class NodeUpdate(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NodeUpdate, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, s_vecs, pred_vecs, o_vecs):
        t_vecs = torch.cat([s_vecs, pred_vecs, o_vecs],dim=1)
        out = self.linear(t_vecs)
        out = self.relu(out)
        return out
    
class GraphTripleConvNet(nn.Module):
    """ A sequence of scene graph convolution layers  """
    def __init__(self, input_dim, num_layers=5, hidden_dim=512, pooling='avg',
                 mlp_normalization='none'):
        super(GraphTripleConvNet, self).__init__()

        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        gconv_kwargs = {
          'input_dim': input_dim,
          'hidden_dim': hidden_dim,
          'pooling': pooling,
          'mlp_normalization': mlp_normalization,
        }
        for _ in range(self.num_layers):
          self.gconvs.append(GraphBlock(**gconv_kwargs))

    def forward(self, obj_vecs, pred_vecs, edges):
        for i in range(self.num_layers):
          gconv = self.gconvs[i]
          obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
        return obj_vecs, pred_vecs