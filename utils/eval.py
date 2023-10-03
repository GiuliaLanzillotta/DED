# Tools for the evaluation of a teacher-student setup 


import numpy as np 

import torch.nn.functional as F

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils import parameters_to_vector


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

def distance_models(teacher, student):
       """Measuring the distance between the models in parameter space"""
       theta_teacher = parameters_to_vector(teacher.parameters()).detach()
       theta_student = parameters_to_vector(student.parameters()).detach()
       return torch.norm(theta_teacher-theta_student).item()