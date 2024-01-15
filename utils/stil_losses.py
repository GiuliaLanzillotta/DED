""" Script implementing different distillation losses."""

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageNet, ImageFolder
from torchvision.models import efficientnet_v2_s, resnet50, ResNet50_Weights, resnet18
import torchvision.transforms as transforms
from utils.nets import DictionaryNet
from utils.kernels import centered_kernal_alignment




def vanilla_distillation(student_out, teacher_out): 
    """ Classic distillation comparing the last layer activations in the networks. 
        student_out: BxO tensor
        teacher_out: BxO tensor 
    """
    return F.mse_loss(student_out, teacher_out)

def topK_distillation(student_out, teacher_out, K):
    """ Distillation comparing the top-K last layer activations, 
    where the ordering is based on the [teacher's network predictions ?].
    """
    _, topk_idx = torch.sort(teacher_out.clone().detach(), descending=True)
    teach_topk = torch.gather(teacher_out, dim = 1, index = topk_idx)[:,:K]
    stud_topk = torch.gather(student_out, dim = 1, index = topk_idx)[:,:K]
    return F.mse_loss(stud_topk, teach_topk)


def inner_distillation(student_net:DictionaryNet, teacher_activations, x):
    """ Distillation comparing the outputs of all network's layers. """
    loss = 0.
    count = 0
    for n, act_s in teacher_activations.items(): 
        if n=='fc': x = torch.flatten(x, 1)
        act_t = teacher_activations[n] # it will throw an error if the two networks are different
        act_s = student_net(x, name=n)
        loss += F.mse_loss(act_s, act_t)
        count+=1
        x = act_t
    return loss/count #average across depth


def deep_inner_distillation(student_net, teacher_activations):
    """ Distillation comparing the outputs of all networks' layers. 
    We single out specific layers and force the student to match the teacher output given the same input."""
    loss = 0.
    count = 0
    for (n, m_s) in student_net.named_modules():
        if sum([p.numel() for p in m_s.parameters()])>0 and len(list(m_s.children()))==0: 
            act_t = teacher_activations[n+"_out"]
            x = teacher_activations[n+"_in"]
            act_s = m_s(x)
            loss += F.mse_loss(act_s, act_t)
            count+=1
    return loss/count #average across depth


def mixed_inner_distillation_free(student_activations, teacher_activations, gamma=1.0):
    """ Feature wise distillation on the last layer, kernel distillation on all the previous ones.
    gamma is the mixing coefficient, defining the weight of the kernel distillation """
    stud_fc = {'fc':student_activations.pop('fc')}
    teach_fc = {'fc':teacher_activations.pop('fc')}
    l_fc, _ = inner_distillation_free(stud_fc, teach_fc)
    l, c = kernel_inner_distillation_free(student_activations, teacher_activations)
    return (l*c*gamma + l_fc)/(c+1), c+1

def fkd(A_s, A_t, lamda_fr, lamda_fk):
    """ Feature kernel distillation between student and teacher. 
        - A_s: BxDs matrix with student features
        - A_t: BxDt matrix with teacher features """
    B = A_s.size(0)
    assert A_t.size(0) == B, "The same number of samples must be provided from teacher and student"
    # normalising first 
    features_norm = A_s.norm(dim=1).view(B,1)
    A_s = torch.div(A_s, features_norm)
    with torch.no_grad():
        A_t = torch.div(A_t, A_t.norm(dim=1).view(B,1))
    # computing kernels 
    with torch.no_grad():
        ker_t = torch.matmul(A_t, A_t.T)
    ker_s = torch.matmul(A_s, A_s.T)

    kernel_loss = ((ker_t - ker_s)**2).sum()/(B*(B-1))
    features_regulariser = features_norm.mean()
    loss = lamda_fk*kernel_loss + lamda_fr*features_regulariser
    return loss, kernel_loss, features_regulariser

def cka_loss(A_s, A_t, power=1):
    """ Feature kernel distillation between student and teacher. 
        - A_s: BxDs matrix with student features
        - A_t: BxDt matrix with teacher features """
    B = A_s.size(0)
    assert A_t.size(0) == B, "The same number of samples must be provided from teacher and student"
    
    with torch.no_grad():
        ker_t = torch.matmul(A_t, A_t.T)
    ker_s = torch.matmul(A_s, A_s.T)

    loss = 1 - centered_kernal_alignment(ker_s, ker_t)
    return loss

def cka_loss(ker_s, ker_t):
    """ Feature kernel distillation between student and teacher. 
        - ker_s: BxB kernel matrix for student
        - ker_t: BxB kernel matrix for teacher"""
    loss = 1 - centered_kernal_alignment(ker_s, ker_t)
    return loss

def features_mse(phi_s, phi_t):
    """Loss for feature distillation."""
    assert phi_s.shape == phi_t.shape, "Student and Teacher features must be matching for feature distillation."
    return F.mse_loss(phi_s, phi_t)

def kernel_inner_distillation_free(student_activations, teacher_activations):
    """ Distillation comparing the feature kernels of the layers recorded. """
    loss = 0.
    count = 0
    for n, act_s in teacher_activations.items(): 
        act_t = teacher_activations[n] # it will throw an error if the two networks are different
        act_s = student_activations[n]
        if not count: B = act_t.shape[0] # register batch size at the beginning
        At = act_t.view(B,-1)
        At = torch.div(At, At.norm(dim=1).view(B,1))
        ker_t = torch.matmul(At, At.T)
        As = act_s.view(B,-1)
        As = torch.div(As, As.norm(dim=1).view(B,1))
        ker_s = torch.matmul(As, As.T)
        loss += F.mse_loss(ker_t, ker_s)
        count+=1
    return loss/count, count #average across depth

def inner_distillation_free(student_activations, teacher_activations):
    """ Distillation comparing the features of the layers recorded. """
    loss = 0.
    count = 0
    for n, act_s in teacher_activations.items(): 
        act_t = teacher_activations[n] # it will throw an error if the two networks are different
        act_s = student_activations[n]
        loss += F.mse_loss(act_t, act_s)
        count+=1
    return loss/count, count #average across depth

def topbottomK_distillation(student_out, teacher_out, K):
    """ Distillation comparing the top-K and bottom-K last layer activations, 
    where the ordering is based on the teacher's network predictions.
    """
    k = K//2 # we use half for each side
    _, ordered_idx = torch.sort(teacher_out.clone().detach(), descending=True)
    teach_topk = torch.gather(teacher_out, dim = 1, index = ordered_idx)[:,:k]
    teach_bottomk = torch.gather(teacher_out, dim = 1, index = ordered_idx)[:,-k:]
    stud_topk = torch.gather(student_out, dim = 1, index = ordered_idx)[:,:k]
    stud_bottomk = torch.gather(student_out, dim = 1, index = ordered_idx)[:,-k:]
    return 0.5*F.mse_loss(stud_topk, teach_topk) + 0.5*F.mse_loss(stud_bottomk, teach_bottomk)


def randomK_distillation(student_out, teacher_out, K):
    """ Distillation comparing a random K subset of the last layer activations, 
    where the ordering is based on the teacher's network predictions.
    """
    random_idx = torch.multinomial(torch.ones_like(teacher_out), K)
    teach_randk = torch.gather(teacher_out, dim = 1, index = random_idx)
    stud_randk = torch.gather(student_out, dim = 1, index = random_idx)
    return F.mse_loss(stud_randk, teach_randk)

