"""Utils to support network structures ..."""
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

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


def register_hooks_layer(network:nn.Module, layer_name):
    """ Register forward hooks for a given network layer"""
    if not hasattr(network, 'activation'): network.activation = {}
    def get_activation(name):
            def hook(model, input, output):
                    network.activation[name] = output
            return hook
    print(f'Registering hook for {layer_name}')
    m = getattr(network, layer_name)
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
    m = getattr(network, layer_name)
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