"""Utilities for kernel measures"""

import torch
import numpy as np
from utils.nets import feature_wrapper
from utils.status import ProgressBar
from torch.func import functional_call, vmap, vjp, jvp, jacrev

def HSIC(K,L):
    """Computes Hilbert-Schmidt Independence Criterion of matrices K and L"""
    n = K.shape[0]
    H = torch.eye(n) - torch.ones((n,n))/n 
    H = H.to(K.device)
    KH = torch.matmul(K,H)
    LH = torch.matmul(L,H)
    HSIC = torch.trace(torch.matmul(KH,LH))/((n-1)**2)
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

# ***************** NTK ***************
# resources: pytorch tutorial https://pytorch.org/tutorials/intermediate/neural_tangent_kernels.html 
def get_params(net):
    params = {k: v.detach() for k, v in net.named_parameters()}
    return params

def get_fnet_func(net):
    def fnet_single(params, x):
        return functional_call(net, params, (x.unsqueeze(0),)).squeeze(0)
    return fnet_single

def empirical_ntk_jacobian_contraction(fnet_single, params, x1, x2, compute='full'):
    # Compute J(x1)
    jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = jac1.values()
    jac1 = [j.flatten(2) for j in jac1]

    # Compute J(x2)
    jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = jac2.values()
    jac2 = [j.flatten(2) for j in jac2]

    # Compute J(x1) @ J(x2).T
    einsum_expr = None
    if compute == 'full':
        einsum_expr = 'Naf,Mbf->NMab'
    elif compute == 'trace':
        einsum_expr = 'Naf,Maf->NM'
    elif compute == 'diagonal':
        einsum_expr = 'Naf,Maf->NMa'
    else:
        assert False

    result = torch.stack([torch.einsum(einsum_expr, j1, j2) for j1, j2 in zip(jac1, jac2)])
    result = result.sum(0)
    return result

def empirical_ntk_ntk_vps(func, params, x1, x2, compute='full'):
    def get_ntk(x1, x2):
        def func_x1(params):
            return func(params, x1)

        def func_x2(params):
            return func(params, x2)

        output, vjp_fn = vjp(func_x1, params)

        def get_ntk_slice(vec):
            # This computes ``vec @ J(x2).T``
            # `vec` is some unit vector (a single slice of the Identity matrix)
            vjps = vjp_fn(vec)
            # This computes ``J(X1) @ vjps``
            _, jvps = jvp(func_x2, (params,), vjps)
            return jvps

        # Here's our identity matrix
        basis = torch.eye(output.numel(), dtype=output.dtype, device=output.device).view(output.numel(), -1)
        return vmap(get_ntk_slice)(basis)

    # ``get_ntk(x1, x2)`` computes the NTK for a single data point x1, x2
    # Since the x1, x2 inputs to ``empirical_ntk_ntk_vps`` are batched,
    # we actually wish to compute the NTK between every pair of data points
    # between {x1} and {x2}. That's what the ``vmaps`` here do.
    result = vmap(vmap(get_ntk, (None, 0)), (0, None))(x1, x2)

    if compute == 'full':
        return result
    if compute == 'trace':
        return torch.einsum('NMKK->NM', result)
    if compute == 'diagonal':
        return torch.einsum('NMKK->NMK', result)

def get_ntk(data_loader, device, model, num_batches=100, silent=False, mode="full"):
    # running estimate of the outer products and mean
    total=0
    if not silent: progress_bar = ProgressBar(verbose=True)

    X = [] # we collect all the gradients in a matrix NxO

    for i, data in enumerate(data_loader):
        if i==num_batches: break 
        inputs, _ = data
        inputs = inputs.to(device)
        B = inputs.size(0)
        X.append(inputs)
        total+=B
                
        if not silent: 
            progress_bar.prog(i, len(data_loader), -1, 'Collecting inputs', i/(min(len(data_loader),num_batches)))  
    
    X = torch.vstack(X).view((total,)+inputs.shape[1:]).to(device)
    ntk = empirical_ntk_jacobian_contraction(get_fnet_func(model), get_params(model), X, X, compute=mode)
    #ntk = empirical_ntk_ntk_vps(get_fnet_func(model), get_params(model), X, X)
    return ntk