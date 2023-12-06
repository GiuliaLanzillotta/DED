"""Utils to support network structures ..."""
import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, resnet50, googlenet, efficientnet_b0, MobileNetV3
import torchvision

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
    
    return model


class CNN(nn.Module):
     
     def __init__(self, c, num_classes, use_batch_norm=True) -> None:
          super().__init__()
          self.features = nn.Sequential(
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
            nn.MaxPool2d(4),
            Flatten()
        )
          self.head = nn.Linear(c*8, num_classes, bias=True)
    
     def get_features(self, x):
        return self.features(x)

     def forward(self, x):
        output = self.get_features(x)
        output = self.head(output)

        return output
        

def count_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params





def get_features(x, model, model_type):
     """ Returns the feature representation of the given input for the given model.
     Model types supported: """
     if model_type=="resnet18":
          phi = model.features(x)
     

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)
    
    def get_features(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        return output

    def forward(self, x):
        output = self.get_features(x)
        output = self.fc(output)

        return output

def resnet18(num_classes):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)



def feature_wrapper(model):
     """Defines a 'get_feature' function in the model based on the model class."""

     if isinstance(model, ResNet):
          #there's already a feature function
          return model 
     
     if isinstance(model, MobileNetV3):
        def get_features(self, x):
               x = self.features(x)
               x = self.avgpool(x)
               x = torch.flatten(x, 1)
               return x
        model.get_features = get_features
        return model
        
     if isinstance(model, torchvision.models.ResNet):
          def get_features(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                return x
          model.get_features = get_features
          return model
     
     if isinstance(model, CNN):
          return model
     
     raise NotImplementedError("the selected model has no 'get_features' method")