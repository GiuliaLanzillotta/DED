"""Utilities for kernel measures"""

import torch
import numpy as np
from utils.nets import feature_wrapper
from utils.status import ProgressBar


def HSIC(K,L):
    """Computes Hilbert-Schmidt Independence Criterion of matrices K and L"""
    n = K.shape[0]
    H = torch.eye(n) - torch.ones((n,n))/n 
    H = H.to(K.device)
    KH = torch.matmul(K,H)
    LH = torch.matmul(L,H)
    HSIC = torch.trace(torch.matmul(KH,LH))/(n-1)**2
    return HSIC

def centered_kernal_alignment(K,L):
    """ computes CKA index based on two empirical Kernel matrices.
    See https://arxiv.org/pdf/1905.00414.pdf for more info."""
    CKA = HSIC(K,L) / torch.sqrt(HSIC(K,K)*HSIC(L,L))
    return CKA


def features_alignment(F1,F2):
    """ computes Feature Alignment between two feature functions.
    F1 , F2 of shape NxD"""
    N = F1.shape[0]
    assert F2.shape[0] == N, "features matrices must be of the same dimensionality"
    A = torch.trace(torch.matmul(F1, F2.T))/((N-1)**2)
    norm1 = torch.trace(torch.matmul(F1, F1.T))/((N-1)**2)
    norm2 = torch.trace(torch.matmul(F2, F2.T))/((N-1)**2)
    FA = A / torch.sqrt(norm1 * norm2)
    return FA

def get_features(data_loader, model, device, num_batches=100, silent=False, center=False):
    # running estimate of the outer products and mean
    total=0
    if not silent: progress_bar = ProgressBar(verbose=True)

    model = feature_wrapper(model) # adding 'get_features' function
    features = [] # we collect all the features in a matrix
    for i, data in enumerate(data_loader):
        if i==num_batches: break 
        with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                B = inputs.size(0)
                phi = model.get_features(inputs)
                # normalising the features to have norm one 
                phi = phi.view(B,-1)
                phi = torch.div(phi, phi.norm(dim=1).view(B,1))

                features.append(phi)

                total += B
                
        if not silent: progress_bar.prog(i, len(data_loader), -1, 'Collecting features', i/(min(len(data_loader),num_batches)))  
    
    F = phi.size(1) # feature dimensionality
    features = torch.vstack(features).view(total, F)
    if center:
        features = features - features.mean(dim=1).view(-1,1)

    return features

def compute_empirical_kernel(data_loader, model, device, num_batches=100, silent=False, center=False):
    # running estimate of the outer products and mean
    features = get_features(data_loader, model, device, num_batches, center=center)
    kernel_matrix = torch.matmul(features, features.T)
    return kernel_matrix

def compute_empirical_kernel_from_features(features):
    """ Features is a NxD torch Tensor with the layer features."""
    # running estimate of the outer products and mean
    kernel_matrix = torch.matmul(features, features.T)
    return kernel_matrix