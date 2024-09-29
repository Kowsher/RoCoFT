import torch
import torch.nn as nn

class Pruner_random(nn.Module):
    def __init__(self, sparsity, descending):
        """
        Initializes the Pruner class.
        
        Args:
            sparsity (float): The desired sparsity, a value between 0 and 1. 
                              This determines how many input dimensions will be pruned.
        """
        super(Pruner_random, self).__init__()
        assert 0 <= sparsity <= 1, "Sparsity must be between 0 and 1"
        self.sparsity = sparsity
        if descending == False:
            self.descending=True
        else:
            self.descending=False

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
        #metric_sum = metric.sum(dim=1)
        # Flatten the weights to easily sort them
        flat_weights =metric.view(-1)

        # Calculate the number of elements corresponding to the top 10%
        k = int(flat_weights.size(0) * self.sparsity)

        # Get the top-k values (largest ones)
        top_k_values, _ = torch.topk(flat_weights, k, largest=self.descending)

        # Create a mask where values greater than or equal to the smallest value in top-k are True
        threshold = top_k_values[-1]
        random_mask = metric >= threshold
        
        return random_mask

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



class Pruner_row(nn.Module):
    def __init__(self, rank, descending):
        """
        Initializes the Pruner class.
        
        Args:
            rank (int): The rank of the matrix to be pruned.
        """
        super(Pruner_row, self).__init__()
        assert 0 <= rank,  "rank must be between grater than 0"
        self.rank = rank
        self.descending = descending

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
        X =  X.norm(p=2, dim=0).view(-1, 1)
        W = W.abs() * X
     
        # Calculate the pruning metric: abs(W) * norm of X
        metric_sum = torch.sum(W, dim=1)  # Metric based on W and X
        
        # Sum the metric along the input dimension (C_in) to get a shape of (C_out,)
       

        # Sort the summed metric along the output dimension (C_out)
        _, sorted_idx = torch.sort(metric_sum, dim=0, descending=self.descending)

  
        
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

class Pruner_column(nn.Module):
    def __init__(self, rank, descending):
        """
        Initializes the Pruner class.
        
        Args:
            rank (int): The rank of the matrix to be pruned.
        """
        super(Pruner_column, self).__init__()
        assert 0 <= rank,  "rank must be between grater than 0"
        self.rank = rank
        self.descending = descending

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
        #print('chk', X.shape, W.shape)
        X =  X.norm(p=2, dim=0)
        #print(X.shape)
        W =  W.abs() * X
        metric_sum = torch.sum(W, dim=0)
        

        # Calculate the pruning metric: abs(W) * norm of X
        #metric_sum = W * X  # Metric based on W and X

        # Sum the metric along the input dimension (C_in) to get a shape of (C_out,)
       

        # Sort the summed metric along the output dimension (C_out)
        _, sorted_idx = torch.sort(metric_sum,dim=0, descending=self.descending)

  
        
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
