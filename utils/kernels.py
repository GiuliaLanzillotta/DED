"""Utilities for kernel measures"""

import torch
import numpy as np
from utils.nets import feature_wrapper
from utils.status import ProgressBar


def HSIC(K,L):
    """Computes Hilbert-Schmidt Independence Criterion of matrices K and L"""
    n = K.shape[0]
    H = np.eye(n) - np.ones((n,n))/n 
    KH = np.matmul(K,H)
    LH = np.matmul(L,H)
    HSIC = np.trace(np.matmul(KH,LH))/(n-1)**2
    return HSIC

def centered_kernal_alignment(K,L):
    """ computes CKA index based on two empirical Kernel matrices.
    See https://arxiv.org/pdf/1905.00414.pdf for more info."""
    CKA = HSIC(K,L) / np.sqrt(HSIC(K,K)*HSIC(L,L))
    return CKA


def compute_empirical_kernel(data_loader, model, device, num_batches=100):
    # running estimate of the outer products and mean
    total=0
    progress_bar = ProgressBar(verbose=True)

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
                
        progress_bar.prog(i, len(data_loader), -1, 'Collecting features', i/(min(len(data_loader),num_batches)))  
    
    F = phi.size(1) # feature dimensionality
    features = torch.vstack(features).view(total, F).cpu().numpy()

    kernel_matrix = np.matmul(features, features.T)
    return kernel_matrix