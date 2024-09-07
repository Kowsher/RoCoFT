import torch
import torch.nn as nn
import random

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