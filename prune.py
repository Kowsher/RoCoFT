import torch
import torch.nn as nn

class Pruner_random(nn.Module):
    def __init__(self, sparsity):
        """
        Initializes the Pruner class.
        
        Args:
            sparsity (float): The desired sparsity, a value between 0 and 1. 
                              This determines how many input dimensions will be pruned.
        """
        super(Pruner_random, self).__init__()
        assert 0 <= sparsity <= 1, "Sparsity must be between 0 and 1"
        self.sparsity = sparsity

    def prune(self, W, X):
        """
        Prunes the weight matrix W based on the input X and desired sparsity.
        
        Args:
            W (torch.Tensor): Weight matrix of shape (C_out, C_in).
            X (torch.Tensor): Input matrix of shape (N, L, C_in).
            
        Returns:
            torch.Tensor: The indices of the pruned weights.
        """
        # Reshape input X from (N, L, C_in) to (N * L, C_in)
        N, L, C_in = X.shape
        X = X.view(N * L, C_in)

        # Calculate the pruning metric: abs(W) * norm of X
        metric = W.abs() * X.norm(p=2, dim=0)  # Metric based on W and X

        # Sum the metric along the input dimension (C_in) to get a shape of (C_out,)
        metric_sum = metric.sum(dim=1)

        # Sort the summed metric along the output dimension (C_out)
        _, sorted_idx = torch.sort(metric_sum, dim=0)

        # Compute the number of elements to prune based on the desired sparsity
        num_pruned = int(W.shape[0] * self.sparsity)
        
        # Select indices to prune
        pruned_idx = sorted_idx[:num_pruned]
        
        return pruned_idx

    def forward(self, W, X):
        """
        Forward pass method for pruning the weights during the forward pass of a model.
        
        Args:
            W (torch.Tensor): Weight matrix of shape (C_out, C_in).
            X (torch.Tensor): Input matrix of shape (N, L, C_in).
        
        Returns:
            torch.Tensor: The indices of the pruned weights.
        """
        return self.prune(W, X)


class Pruner(nn.Module):
    def __init__(self, rank):
        """
        Initializes the Pruner class.
        
        Args:
            rank (int): The rank of the matrix to be pruned.
        """
        super(Pruner, self).__init__()
        assert 0 <= rank,  "rank must be between grater than 0"
        self.rank = rank

    def prune(self, W, X):
        """
        Prunes the weight matrix W based on the input X and desired sparsity.
        
        Args:
            W (torch.Tensor): Weight matrix of shape (C_out, C_in).
            X (torch.Tensor): Input matrix of shape (N, L, C_in).
            
        Returns:
            torch.Tensor: The indices of the pruned weights.
        """
        # Reshape input X from (N, L, C_in) to (N * L, C_in)
        N, L, C_in = X.shape
        X = X.view(N * L, C_in)

        # Calculate the pruning metric: abs(W) * norm of X
        metric = W.abs() * X.norm(p=2, dim=0)  # Metric based on W and X

        # Sum the metric along the input dimension (C_in) to get a shape of (C_out,)
        metric_sum = metric.sum(dim=1)

        # Sort the summed metric along the output dimension (C_out)
        _, sorted_idx = torch.sort(metric_sum, dim=0)

  
        
        # Select indices to prune
        pruned_idx = sorted_idx[:self.rank]
        
        return pruned_idx

    def forward(self, W, X):
        """
        Forward pass method for pruning the weights during the forward pass of a model.
        
        Args:
            W (torch.Tensor): Weight matrix of shape (C_out, C_in).
            X (torch.Tensor): Input matrix of shape (N, L, C_in).
        
        Returns:
            torch.Tensor: The indices of the pruned weights.
        """
        return self.prune(W, X)
