import torch

def avg_siblings(x, sibling_order, h_summary_size=2, w_summary_size=2):
    """ Average the siblings of a node in the tree.
    Args:
        x: torch.Tensor of shape (B, H, W, R)
        sibling_order: int, the order of the siblings to average
        h_summary_size: int, the size of the summary in the height dimension
        w_summary_size: int, the size of the summary in the width dimension
    Returns:
        x: torch.Tensor of shape (B, H, W, R)
    """
    if sibling_order == 0:
        return x
    B, H, W, R = x.size()

    # Ensure sibling_order is a tensor and cast it to int64 for exponentiation
    sibling_order = torch.as_tensor(sibling_order, dtype=torch.int64, device=x.device)

    # Use tensor operations to avoid breaking TorchDynamo graph
    h_num_sum = torch.pow(torch.tensor(h_summary_size, dtype=torch.int64, device=x.device), sibling_order)
    w_num_sum = torch.pow(torch.tensor(w_summary_size, dtype=torch.int64, device=x.device), sibling_order)

    h_n_splits = H // h_num_sum
    w_n_splits = W // w_num_sum

    assert h_n_splits * h_num_sum == H, (
        f"H={H} not divisible by {h_num_sum}, might cause zero-size or shape mismatch!"
    )
    assert w_n_splits * w_num_sum == W, (
        f"W={W} not divisible by {w_num_sum}, might cause zero-size or shape mismatch!"
    )

    x_temp = x.reshape(B, h_n_splits, h_num_sum, w_n_splits, w_num_sum, R)
    x_temp = x_temp.mean(dim=[2, 4])  # B, h_n_splits, w_n_splits, R
    x_temp = torch.repeat_interleave(x_temp, repeats=h_num_sum, dim=1)
    return torch.repeat_interleave(x_temp, repeats=w_num_sum, dim=2)
