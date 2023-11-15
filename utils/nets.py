"""Utils to support network structures ..."""
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.nn.functional as F


class LinearNet(nn.Module):
     
     def __init__(self, dim_in, dim_out, **kwargs) -> None:
          super().__init__()
          self.dim_in, self.dim_out = dim_in, dim_out
          self.layer = nn.Linear(dim_in, dim_out, bias=kwargs.get('bias',True))

     def forward(self, x): 
          return self.layer(x)
     
     def predict(self, x):
          return self.forward(x)



class DictionaryNet(nn.Module):

    def __init__(self, network, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleDict({n:m for (n,m) in network.named_modules() if sum([p.numel() for p in m.parameters()])>0})

    def module_forward(self, x, name:str): 
        return self.layers[name](x)

    def net_forward(self, x): 
        for n,m in self.layers.items(): 
            if n=='fc': x = torch.flatten(x, 1) #note: this requires a specific architecture configuration
            x = m(x)
        return x  

    def forward(self, x, name=None):
         if name is None: 
              return self.net_forward(x)
         return self.module_forward(x, name)

def unroll_layer_hierarchy(model, layer_name):
    """ It returns an internal layer of the network by following the hierarchical structure.
    takes a layer name, say layer1.0.conv1 and separates 
    into its hierarchical components [layer1,0,conv1]. """
    hierarchy = layer_name.split('.')
    m = model
    for level in hierarchy: 
        m = getattr(m, level)
    return m

def register_hooks_layer(network:nn.Module, layer_name):
    """ Register forward hooks for a given network layer"""
    if not hasattr(network, 'activation'): network.activation = {}
    def get_activation(name):
            def hook(model, input, output):
                    network.activation[name] = output
            return hook
    print(f'Registering hook for {layer_name}')
    #m = getattr(network, layer_name)
    m = unroll_layer_hierarchy(network, layer_name)
    m.register_forward_hook(get_activation(layer_name))
    return network
      

def register_hooks_layer_input_output(network:nn.Module, layer_name):
    """ Register forward hooks for a given network layer"""
    if not hasattr(network, 'activation'): network.activation = {}
    def get_activation(name):
            def hook(model, input, output):
                    network.activation[name+"_in"] = input[0]
                    network.activation[name+"_out"] = output
            return hook
    print(f'Registering hook for {layer_name}')
    m = unroll_layer_hierarchy(network, layer_name)
    m.register_forward_hook(get_activation(layer_name))
    return network

def register_module_hooks_network(model:nn.Module, N_BLOCKS:int):
    """ Register forward hooks for all the network blocks (children)"""
    tot = 0
    for n,m in reversed(list(model.named_children())):
            if sum([p.numel() for p in m.parameters()])>0 and tot<N_BLOCKS:
                register_hooks_layer(model, n)
                tot+=1
    return model

def register_module_hooks_network_deep(model:nn.Module, parallel=True):
    """ Registers forward hooks for all the parametric network layers (conv, BN, linear).
    If 'parallel' is switched on, then the 'input-output' activations will be saved."""
    # registering forward hooks in the teacher network
    for n,m in model.named_modules():
            if sum([p.numel() for p in m.parameters()])>0 and len(list(m.children()))==0: # we take the individual layers
                    if parallel: register_hooks_layer_input_output(model, n)
                    else: register_hooks_layer(model, n)
    return model



class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), x.size(1))

def make_cnn(c, num_classes, use_batch_norm):
    ''' Returns a 5-layer CNN with width parameter c. '''
    model= nn.Sequential(
        # Layer 0
        nn.Conv2d(3, c, kernel_size=3, stride=1,
                  padding=1, bias=True),
        nn.BatchNorm2d(c) if use_batch_norm else nn.Identity(),
        nn.ReLU(),

        # Layer 1
        nn.Conv2d(c, c*2, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c*2) if use_batch_norm else nn.Identity(),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 2
        nn.Conv2d(c*2, c*4, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c*4) if use_batch_norm else nn.Identity(),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 3
        nn.Conv2d(c*4, c*8, kernel_size=3,
                  stride=1, padding=1, bias=True),
        nn.BatchNorm2d(c*8) if use_batch_norm else nn.Identity(),
        nn.ReLU(),
        nn.MaxPool2d(2),

        # Layer 4
        nn.MaxPool2d(4),
        Flatten(),
        nn.Linear(c*8, num_classes, bias=True)
    )
    
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"CNN made with {params} parameters")

    return model

