import torch

def avg_siblings(x,sibling_order, h_summary_size=1, w_summary_size=1):
    """ Average the siblings of a node in the tree.
    Args:
        x: torch.Tensor of shape (B, H, W, R)
        sibling_order: int, the order of the siblings to average
        h_summary_size: int, the size of the summary in the height dimension
        w_summary_size: int, the size of the summary in the width dimension
        Returns:
        x: torch.Tensor of shape (B, H, W, R)
        """
    
    B, H, W, R = x.size()
    
    h_num_sum = h_summary_size**sibling_order
    w_num_sum = w_summary_size**sibling_order
    h_n_splits = H//h_num_sum
    w_n_splits = W//w_num_sum

    assert isinstance(h_n_splits, int), "h_n_splits must be an integer"
    assert isinstance(w_n_splits, int), "w_n_splits must be an integer"
    h_n_splits = int(h_n_splits)
    w_n_splits = int(w_n_splits)
    
    x_temp =  x.reshape(B, h_n_splits,h_num_sum, w_n_splits, w_num_sum, R)
    x_temp =  x_temp.mean(dim=[2, 4]) # B, h_n_splits, w_n_splits, R
    x_temp = torch.repeat_interleave(x_temp, repeats=h_num_sum, dim=1)
    return torch.repeat_interleave(x_temp, repeats=w_num_sum, dim=2)