# Tools for the evaluation of a teacher-student setup 


import numpy as np 

import torch.nn.functional as F

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils import parameters_to_vector

from utils.kernels import *

def hellinger(p, q):
        """ The Hellinger distance is a bounded metric on the space of 
        probability distributions over a given probability space.
         See https://en.wikipedia.org/wiki/Hellinger_distance
         """
        if torch.is_tensor(p): 
               p = p.detach().cpu().numpy()
               q = q.detach().cpu().numpy()
        return np.linalg.norm(np.sqrt(p) - np.sqrt(q), ord=2, axis=1)/(np.sqrt(2))

def evaluate_regression(model, X, y):
    """Evaluating a simple regression model"""
    pred = model.predict(X)
    if torch.is_tensor(X):
          return F.mse_loss(pred, y).item()
    return np.mean((pred-y)**2)

def evaluate_classification(model, X, y):
    """Evaluating a simple regression model"""
    pred = model.predict(X)
    if torch.is_tensor(X):
          y = y.detach().cpu().numpy()
          pred = pred.detach().cpu().numpy()
    if len(pred.shape)>1: #predicting probabilities
          pred = np.max(pred, 1)
    correct = np.sum(pred == y)
    return correct/X.shape[0]

def instant_eval(model, inputs, labels, device):
       """ simply evaluating the model on the given batch (usable on tensors and torch models)"""
       status = model.training
       model.eval()
       with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                correct = torch.sum(pred == labels).item()
                total = labels.shape[0]               
       acc=(correct / total) * 100
       model.train(status)
       return acc

def instant_function_distance(student, teacher, inputs, labels, device):
       """Evaluating distance in the outputs between teacher and student models"""
       status = student.training
       student.eval()
       teacher.eval()
       with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)
                out_s = student(inputs)
                out_t = teacher(inputs)
                distance = torch.sum(torch.norm(F.softmax(out_s, dim=1)-F.softmax(out_t, dim=1), dim=1, p=2)).item()
                total = labels.shape[0]               
       distance=(distance / total) 
       student.train(status)
       return distance

def evaluate(model, val_loader, device, num_samples=-1):
    status = model.training
    model.eval()
    if num_samples >0: 
          # we select a subset of the validation dataset to validate on 
          # note: a different random sample is used every time
          random_indices = np.random.choice(range(len(val_loader.dataset)), size=num_samples, replace=False)
          _subset = Subset(val_loader.dataset, random_indices)
          val_loader =  DataLoader(_subset, 
                                   batch_size=val_loader.batch_size, 
                                   shuffle=False, num_workers=4, pin_memory=False)
    correct, total = 0.0, 0.0
    for i,data in enumerate(val_loader):
        with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]
                        
    acc=(correct / total) * 100
    model.train(status)
    return acc


def validation_and_agreement(student, teacher, val_loader, device, num_samples=-1):
        """ Like evaluate, but it also returns the average agreement of student and teacher"""
        status = student.training
        student.eval()
        teacher.eval() # shouldn't be needed
        if num_samples >0: 
                # we select a subset of the validation dataset to validate on 
                # note: a different random sample is used every time
                random_indices = np.random.choice(range(len(val_loader.dataset)), size=num_samples, replace=False)
                _subset = Subset(val_loader.dataset, random_indices)
                val_loader =  DataLoader(_subset, 
                                        batch_size=val_loader.batch_size, 
                                        shuffle=False, num_workers=4, pin_memory=False)
        correct, total, agreement = 0.0, 0.0, 0.0
        for i,data in enumerate(val_loader):
                with torch.no_grad():
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs_s = student(inputs)
                        outputs_t = teacher(inputs)
                        _, pred_s = torch.max(outputs_s.data, 1)
                        _, pred_t = torch.max(outputs_t.data, 1)
                        correct += torch.sum(pred_s == labels).item()
                        agreement += torch.sum(pred_s == pred_t).item()
                        total += labels.shape[0]
                                
        acc=(correct / total) * 100
        agreement = (agreement / total) *100
        student.train(status)
        return acc, agreement

def validation_agreement_function_distance(student, teacher, val_loader, device, num_samples=-1):
        """ Like evaluate, but it also returns the average agreement of student and teacher"""
        status = student.training
        student.eval()
        teacher.eval() # shouldn't be needed
        if num_samples >0: 
                # we select a subset of the validation dataset to validate on 
                # note: a different random sample is used every time
                random_indices = np.random.choice(range(len(val_loader.dataset)), size=num_samples, replace=False)
                _subset = Subset(val_loader.dataset, random_indices)
                val_loader =  DataLoader(_subset, 
                                        batch_size=val_loader.batch_size, 
                                        shuffle=False, num_workers=4, pin_memory=False)
        correct, total, agreement, function_distance = 0.0, 0.0, 0.0, 0.0
        for i,data in enumerate(val_loader):
                with torch.no_grad():
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs_s = student(inputs)
                        outputs_t = teacher(inputs)
                        _, pred_s = torch.max(outputs_s.data, 1)
                        _, pred_t = torch.max(outputs_t.data, 1)
                        correct += torch.sum(pred_s == labels).item()
                        agreement += torch.sum(pred_s == pred_t).item()
                        prob_s = F.softmax(outputs_s, dim=1); prob_t = F.softmax(outputs_t, dim=1)
                        function_distance += np.sum(hellinger(prob_s, prob_t))
                        total += labels.shape[0]
                                
        acc=(correct / total) * 100
        agreement = (agreement / total) *100
        function_distance = (function_distance / total)
        student.train(status)
        return acc, agreement, function_distance


def validation_and_L2distance(student, teacher, val_loader, device, num_samples=-1, use_teacher=True):
        """ Like evaluate, but it also returns the average agreement of student and teacher"""
        status = student.training
        student.eval()
        teacher.eval() # shouldn't be needed
        if num_samples >0: 
                # we select a subset of the validation dataset to validate on 
                # note: a different random sample is used every time
                random_indices = np.random.choice(range(len(val_loader.dataset)), size=num_samples, replace=False)
                _subset = Subset(val_loader.dataset, random_indices)
                val_loader =  DataLoader(_subset, 
                                        batch_size=val_loader.batch_size, 
                                        shuffle=False, num_workers=4, pin_memory=False)
        correct, total, l2distance, hdistance = 0.0, 0.0, 0.0, 0.0
        for i,data in enumerate(val_loader):
                with torch.no_grad():
                        inputs, labels = data
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs_s = student(inputs)
                        prob_s = F.softmax(outputs_s, dim=1)
                        if use_teacher: 
                               outputs_t = teacher(inputs)
                               prob_t = F.softmax(outputs_t, dim=1)
                        correct += torch.sum(torch.max(outputs_s.data, 1)[1] == labels).item()
                        if use_teacher:
                                l2distance += torch.norm(prob_s-prob_t, dim=1).sum()
                                hdistance += np.sum(hellinger(prob_s, prob_t))
                        else: 
                               labels = F.one_hot(labels, num_classes=outputs_s.size(1)).to(torch.float)
                               l2distance += torch.norm(prob_s - labels, dim=1).sum()
                               hdistance += np.sum(hellinger(prob_s, labels))
                        total += labels.shape[0]
                                
        acc=(correct / total) * 100
        l2distance=l2distance/total
        hdistance=hdistance/total
        student.train(status)
        return acc, l2distance, hdistance

def distance_models(teacher, student):
       """Measuring the distance between the models in parameter space"""
       theta_teacher = parameters_to_vector(teacher.parameters()).detach()
       theta_student = parameters_to_vector(student.parameters()).detach()
       return torch.norm(theta_teacher-theta_student).item()


def evaluate_CKA_teacher(teacher, student, loader, device, batches=10):
       """ evaluates CKA between teacher and student on the features (second to last layer)"""
        
       print("Evaluating CKA ... ") 
       KT = compute_empirical_kernel(loader, teacher, device, batches)
       KS = compute_empirical_kernel(loader, student, device, batches)

       CKA = centered_kernal_alignment(KT,KS).cpu().item()

       return CKA
       