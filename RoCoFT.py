import torch
import torch.nn as nn
import random
import prune

class row(nn.Module):
    def __init__(self, F, rank=1, bias=True):
        super(row, self).__init__()
        F.eval()

        assert 0 <= rank, f"rank must be between 0 and {F.weight.shape[0]}"

        total_weights = F.weight.shape[0]
        k = min(rank, total_weights)

        # Trainable weights
        self.trainable_weight = nn.Parameter(F.weight[:k, :].clone())
        
        # Non-trainable weights (detached and moved to a buffer to save memory)
        self.register_buffer('non_trainable_weight', F.weight[k:, :].clone().detach())
        #self.trainable_weight.register_hook(lambda grad: grad.to(self.trainable_weight.device))
        self.non_trainable_weight.grad = None

        # Bias management
        if F.bias is not None:
            self.bias = nn.Parameter(F.bias.clone().detach(), requires_grad = bias)
        else:
            self.bias = None

    def forward(self, x):
        # Concatenate only once during initialization to avoid dynamic tensor creation
        full_weight = torch.cat([self.trainable_weight, self.non_trainable_weight], dim=0)
        out = torch.nn.functional.linear(x, full_weight, self.bias)
        return out


class column(nn.Module):
    def __init__(self, F, rank=1, bias=True):
        super(column, self).__init__()
        F.eval()
        total_weights = F.weight.shape[1]
        
        assert 0 <= rank, f"rank must be between 0 and {F.weight.shape[1]}"
        
        k = min(rank, total_weights)

        # Trainable weights
        self.trainable_weight = nn.Parameter(F.weight[:, :k].clone())
        
        # Non-trainable weights (detached and moved to a buffer to save memory)
        self.register_buffer('non_trainable_weight', F.weight[:, k:].clone().detach())
        #self.trainable_weight.register_hook(lambda grad: grad.to(self.trainable_weight.device))
        self.non_trainable_weight.grad = None

        # Bias management
        if F.bias is not None:
            self.bias = nn.Parameter(F.bias.clone().detach(), requires_grad = bias)
        else:
            self.bias = None

    def forward(self, x):
        # Concatenate only once during initialization to avoid dynamic tensor creation
        full_weight = torch.cat([self.trainable_weight, self.non_trainable_weight], dim=1)
        out = torch.nn.functional.linear(x, full_weight, self.bias)
        return out


class random(nn.Module):
    def __init__(self, F, rank=0.1, bias=True):
        super(random, self).__init__()
        
        # Ensure that rank is a valid probability between 0 and 1
        assert 0 <= rank <= 1, "rank must be a probability between 0 and 1"
        
        # F is a pretrained layer
        F.eval()

        self.weight = nn.ParameterList()

        # Create a random mask to select p% of parameters
        self.random_mask = torch.rand(F.weight.size(), device=F.weight.device) <= rank

        # Clone the weights and apply the mask
        W = F.weight.detach().clone()
        trainable = nn.Parameter(W, requires_grad=True)

        # Register a hook to zero out the gradients for non-selected weights
        trainable.register_hook(lambda grad: grad * self.random_mask)

        self.weight.append(trainable)

        # Managing bias
        if F.bias is not None:
            self.bias = nn.Parameter(F.bias.detach().clone(), requires_grad=bias)
        else:
            self.bias = None

    def forward(self, x):
        out = torch.nn.functional.linear(x, self.weight[0], self.bias)
        return out


def PEFT(model, method, rank, targets=[], bias=True):
    """
    Recursively replaces nn.Linear layers with a custom layer.
    """
    #print(targets, len(targets))
    
    if isinstance(method, str):
        method = globals()[method]
        
    for name, module in model.named_children():
        #print(name, module)

        # Replace nn.Linear with CustomLinear_colum
        if isinstance(module, nn.Linear):
            for param in module.parameters():
                param.requires_grad = False
            #print(name)
            if len(targets)>0:
                if name in targets:#== 'value' or  name == 'query' or  name == 'key': #( name == 'value' or  name == 'query' or  name == 'key')    
                    #print(f'Replacing layer {name}')
                    # Instantiate the custom layer with the current module as argument
                    custom_layer = method(module, rank, bias)
                    setattr(model, name, custom_layer)
            else:
                #print(f'Replacing layer {name}')
                custom_layer = method(module, rank, bias)
                setattr(model, name, custom_layer)                
        elif isinstance(module, (nn.Embedding, nn.LayerNorm)):
            for param in module.parameters():
                param.requires_grad = False
        else:
            # Recursively apply this function to children modules
            PEFT(module, method,rank, targets, bias)



class row_prune(nn.Module):
    def __init__(self, F, rank_indices, bias=True):
        """
        Custom layer to allow fine-tuning only certain rows in a weight matrix, 
        while keeping the rest of the weights non-trainable, and preserving the 
        original position of the weights.
        
        Args:
            F (nn.Linear): The original fully connected layer.
            rank_indices (list or tensor): The indices of the rows in the weight matrix 
                                           that should be trainable.
            bias (bool): Whether to keep the bias trainable.
        """
        super(row_prune, self).__init__()
        F.eval()

        # Convert rank_indices to a tensor if it's a list
        if isinstance(rank_indices, list):
            rank_indices = torch.tensor(rank_indices)

        if F.weight.shape[0]<= len(rank_indices):
            if rank_indices[F.weight.shape[0]-1] > F.weight.shape[0]-1:
                rank_indices = torch.arange(F.weight.shape[0])
            else:
                rank_indices = rank_indices[0:F.weight.shape[0]]

        # Ensure the rank_indices are within valid bounds
        #print(rank_indices, F.weight.shape)
        assert torch.all(rank_indices < F.weight.shape[0]), f"Indices must be within range [0, {F.weight.shape[0] - 1}]"

        total_weights = F.weight.shape[0]
        num_trainable = len(rank_indices)

        # Store the trainable indices and non-trainable indices
        self.rank_indices = rank_indices
        self.non_trainable_indices = torch.tensor(list(set(torch.arange(total_weights).tolist()) - set(rank_indices.tolist())))
        

        # Trainable weights based on rank_indices
        self.trainable_weight = nn.Parameter(F.weight[rank_indices, :].clone())

        # Non-trainable weights: select all the rows not in rank_indices
        if len(self.non_trainable_indices) == 0:
            self.register_buffer('non_trainable_weight', F.weight[len(rank_indices):, :].clone().detach())
        else:    
            self.register_buffer('non_trainable_weight', F.weight[self.non_trainable_indices, :].clone().detach())

        # Bias management
        if F.bias is not None:
            self.bias = nn.Parameter(F.bias.clone().detach(), requires_grad=bias)
        else:
            self.bias = None

        # Save the total shape for reconstruction of the weight matrix
        self.total_weight_shape = F.weight.shape

    def forward(self, x):
        # Initialize the full weight matrix with the correct shape
        full_weight = torch.zeros(self.total_weight_shape, device=x.device)

        # Place the trainable weights in their original positions
        full_weight[self.rank_indices] = self.trainable_weight

        # Place the non-trainable weights in their original positions
        if len(self.non_trainable_indices) > 0:
            #print(self.non_trainable_indices)
            full_weight[self.non_trainable_indices] = self.non_trainable_weight

        # Forward pass using the correctly ordered weights
        out = torch.nn.functional.linear(x, full_weight, self.bias)
        return out


class column_prune(nn.Module):
    def __init__(self, F, rank_indices, bias=True):
        """
        Custom layer to allow fine-tuning only certain columns in a weight matrix, 
        while keeping the rest of the weights non-trainable, and preserving the 
        original position of the weights.
        
        Args:
            F (nn.Linear): The original fully connected layer.
            rank_indices (list or tensor): The indices of the columns in the weight matrix 
                                           that should be trainable.
            bias (bool): Whether to keep the bias trainable.
        """
        super(column_prune, self).__init__()
        F.eval()

        # Convert rank_indices to a tensor if it's a list
        if isinstance(rank_indices, list):
            rank_indices = torch.tensor(rank_indices)

        if 0 not in rank_indices:
            rank_indices[-1] = 0
            

        # Ensure the rank_indices are within valid bounds
        if len(rank_indices) >= F.weight.shape[1]:
            rank_indices = torch.arange(F.weight.shape[1])

        #print(rank_indices, F.weight.shape[1])
        assert torch.all(rank_indices < F.weight.shape[1]), f"Indices must be within range [0, {F.weight.shape[1] - 1}]"

        total_columns = F.weight.shape[1]
        num_trainable = len(rank_indices)

        # Store the trainable indices and non-trainable indices
        self.rank_indices = rank_indices
        self.non_trainable_indices = torch.tensor(list(set(torch.arange(total_columns).tolist()) - set(rank_indices.tolist())))

        # Trainable weights based on rank_indices (select columns)
        self.trainable_weight = nn.Parameter(F.weight[:, rank_indices].clone())

        # Non-trainable weights: select all the columns not in rank_indices
        if len(self.non_trainable_indices) == 0:
            self.register_buffer('non_trainable_weight', torch.zeros(F.weight[:, rank_indices].shape, device=F.weight.device))
        else:
            self.register_buffer('non_trainable_weight', F.weight[:, self.non_trainable_indices].clone().detach())

        # Bias management
        if F.bias is not None:
            self.bias = nn.Parameter(F.bias.clone().detach(), requires_grad=bias)
        else:
            self.bias = None

        # Save the total shape for reconstruction of the weight matrix
        self.total_weight_shape = F.weight.shape

    def forward(self, x):
        # Initialize the full weight matrix with the correct shape
        full_weight = torch.zeros(self.total_weight_shape, device=x.device)

        # Place the trainable weights in their original positions (columns)
        full_weight[:, self.rank_indices] = self.trainable_weight

        # Place the non-trainable weights in their original positions (columns)
        if len(self.non_trainable_indices) > 0:
            full_weight[:, self.non_trainable_indices] = self.non_trainable_weight

        # Forward pass using the correctly ordered weights
        out = torch.nn.functional.linear(x, full_weight, self.bias)
        return out


class random_prune(nn.Module):
    def __init__(self, F, rank, bias=True):
        super(random_prune, self).__init__()
        
        
        # F is a pretrained layer
        F.eval()

        self.weight = nn.ParameterList()

        # Create a random mask to select p% of parameters
        self.random_mask = rank

        # Clone the weights and apply the mask
        W = F.weight.detach().clone()
        trainable = nn.Parameter(W, requires_grad=True)

        # Register a hook to zero out the gradients for non-selected weights
        trainable.register_hook(lambda grad: grad * self.random_mask.to(grad.device))

        self.weight.append(trainable)

        # Managing bias
        if F.bias is not None:
            self.bias = nn.Parameter(F.bias.detach().clone(), requires_grad=bias)
        else:
            self.bias = None

    def forward(self, x):
        out = torch.nn.functional.linear(x, self.weight[0], self.bias)
        return out


def PEFT_prunig_column(model, method, rank, input_data, pruner, targets=[], bias=True):
    """
    Recursively replaces nn.Linear layers with a custom layer, while capturing the output of the original layer.
    The output of each layer is passed as input to the next layer.
    """
   
            
    # Keep track of the input that changes with each layer's output
    current_input = input_data
    
    
    for name, module in model.named_children():
        #print(name)
        # Check for nn.Linear layers to replace
        if isinstance(module, nn.Linear):
            for param in module.parameters():
                param.requires_grad = False
            
            # If the layer matches the targets or is within the desired scope
            if len(targets) > 0:
                if name in targets:
                    # Capture the pretrained output of the layer before replacement
                    with torch.no_grad():
                        pretrained_output = module(current_input)

                    # Replace the layer with the custom method
                    rank_idx=pruner(module, current_input)
                    custom_layer = method(module, rank_idx, bias)
                    setattr(model, name, custom_layer)

                    # The output of the current layer becomes the input to the next
                    current_input = pretrained_output
            else:
                # Capture the pretrained output for all nn.Linear layers

                with torch.no_grad():
                    pretrained_output = module(current_input)
                # Replace the layer with the custom method
                #print('check', module, pretrained_output.shape, module.weight.shape)
                rank_idx=pruner(module.weight, current_input)
                custom_layer = method(module, rank_idx, bias)
                setattr(model, name, custom_layer)

                # Update current_input to be the output of the current layer
                current_input = pretrained_output

        elif isinstance(module, (nn.Embedding, nn.LayerNorm)):
            for param in module.parameters():
                param.requires_grad = False
        else:
            # Recursively apply this function to children modules
            current_input = PEFT_prunig_column(module, method, rank, current_input, pruner, targets, bias)
    return current_input  # Return the final input (or output of the last layer)

def PEFT_prunig_row(model, method, rank, input_data, pruner, targets=[], bias=True):
    """
    Recursively replaces nn.Linear layers with a custom layer, while capturing the output of the original layer.
    The output of each layer is passed as input to the next layer.
    """
   
            
    # Keep track of the input that changes with each layer's output
    current_input = input_data
    
    
    for name, module in model.named_children():
        #print(name)
        # Check for nn.Linear layers to replace
        if isinstance(module, nn.Linear):
            for param in module.parameters():
                param.requires_grad = False
            
            # If the layer matches the targets or is within the desired scope
            if len(targets) > 0:
                if name in targets:
                    # Capture the pretrained output of the layer before replacement
                    with torch.no_grad():
                        pretrained_output = module(current_input)

                    # Replace the layer with the custom method
                    rank_idx=pruner(module, pretrained_output)
                    custom_layer = method(module, rank_idx, bias)
                    setattr(model, name, custom_layer)

                    # The output of the current layer becomes the input to the next
                    current_input = pretrained_output
            else:
                # Capture the pretrained output for all nn.Linear layers

                with torch.no_grad():
                    pretrained_output = module(current_input)
                # Replace the layer with the custom method
                #print('check', module, pretrained_output.shape, module.weight.shape)
                rank_idx=pruner(module.weight, pretrained_output)
                custom_layer = method(module, rank_idx, bias)
                setattr(model, name, custom_layer)

                # Update current_input to be the output of the current layer
                current_input = pretrained_output

        elif isinstance(module, (nn.Embedding, nn.LayerNorm)):
            for param in module.parameters():
                param.requires_grad = False
        else:
            # Recursively apply this function to children modules
            current_input = PEFT_prunig_row(module, method, rank, current_input, pruner, targets, bias)
    return current_input  # Return the final input (or output of the last layer)


def PEFT_prunig_random(model, method, rank, input_data, pruner, targets=[], bias=True):
    """
    Recursively replaces nn.Linear layers with a custom layer, while capturing the output of the original layer.
    The output of each layer is passed as input to the next layer.
    """
   
            
    # Keep track of the input that changes with each layer's output
    current_input = input_data
    
    
    for name, module in model.named_children():
        #print(name)
        # Check for nn.Linear layers to replace
        if isinstance(module, nn.Linear):
            for param in module.parameters():
                param.requires_grad = False
            
            # If the layer matches the targets or is within the desired scope
            if len(targets) > 0:
                if name in targets:
                    # Capture the pretrained output of the layer before replacement
                    with torch.no_grad():
                        pretrained_output = module(current_input)

                    # Replace the layer with the custom method
                    rank_idx=pruner(module, current_input)
                    custom_layer = method(module, rank_idx, bias)
                    setattr(model, name, custom_layer)

                    # The output of the current layer becomes the input to the next
                    current_input = pretrained_output
            else:
                # Capture the pretrained output for all nn.Linear layers

                with torch.no_grad():
                    pretrained_output = module(current_input)
                # Replace the layer with the custom method
                #print('check', module, pretrained_output.shape, module.weight.shape)
                rank_idx=pruner(module.weight, current_input)
                custom_layer = method(module, rank_idx, bias)
                setattr(model, name, custom_layer)

                # Update current_input to be the output of the current layer
                current_input = pretrained_output

        elif isinstance(module, (nn.Embedding, nn.LayerNorm)):
            for param in module.parameters():
                param.requires_grad = False
        else:
            # Recursively apply this function to children modules
            current_input = PEFT_prunig_random(module, method, rank, current_input, pruner, targets, bias)
    return current_input  # Return the final input (or output of the last layer)





def PEFT_prunig(model, method, rank, input_data, targets=[], bias=True, descending=False):

    if method == 'column':
        pruner = prune.Pruner_column(rank, descending)
        if isinstance(method, str):
            method = globals()[method+'_prune']
            _ = PEFT_prunig_column(model, method, rank, input_data, pruner, targets, bias=True)
    elif method == 'row':
        pruner = prune.Pruner_row(rank, descending)
        if isinstance(method, str):
            method = globals()[method+'_prune']
            _ = PEFT_prunig_row(model, method, rank, input_data, pruner, targets, bias=True)   
    
    elif method == 'random':
        pruner = prune.Pruner_random(rank, descending)
        if isinstance(method, str):
            method = globals()[method+'_prune']
            _ = PEFT_prunig_random(model, method, rank, input_data, pruner, targets, bias=True)   
           
