"""Utils to support network structures ..."""
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

class DictionaryNet(nn.Module):

    def __init__(self, network, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.ModuleDict({n:m for (n,m) in network.named_children()})

    def module_forward(self, x, name:str): 
        return self.layers[name](x)

    def net_forward(self, x): 
        for n,m in self.layers.items(): 
            if n=='fc': x = torch.flatten(x, 1)
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

def register_module_hooks_network(model:nn.Module, N_BLOCKS:int):
    """ Register forward hooks for all the network blocks (children)"""
    tot = 0
    for n,m in reversed(list(model.named_children())):
            if sum([p.numel() for p in m.parameters()])>0 and tot<N_BLOCKS:
                register_hooks_layer(model, n)
                tot+=1
    return model

def register_module_hooks_network_deep(model:nn.Module):
    """ Registers forward hooks for all the parametric network layers (conv, BN, linear)"""
    # registering forward hooks in the teacher network
    for n,m in model.named_modules():
            if isinstance(m, nn.Conv2d) or \
                    isinstance(m, nn.BatchNorm2d) or \
                            isinstance(m, nn.Linear):
                    register_hooks_layer(model, n)
    return model