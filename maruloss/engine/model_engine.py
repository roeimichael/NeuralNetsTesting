# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 11:38:37 2022

@author: imargolin
"""

# Import 
from torch import nn

from torch.optim import Adam # type: ignore
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F # type: ignore
import torch # type: ignore
from tqdm import tqdm # type: ignore
from torch.optim import lr_scheduler # type: ignore
from sklearn.metrics import confusion_matrix
import numpy as np
import copy
from torchmetrics import F1Score
import mlflow
from  mlflow.tracking import MlflowClient



print(f"loaded {__name__}!")


class LRScheduler():
    def __init__(self, init_lr=1.0e-4, lr_decay_epoch=10, 
                 lr_decay_factor = 0.9):

        self.init_lr = init_lr
        self.lr_decay_epoch = lr_decay_epoch
        self.lr_decay_factor = lr_decay_factor

    def __call__(self, optimizer, epoch):
        '''Decay learning rate by a factor every lr_decay_epoch epochs.'''
        lr = self.init_lr * (self.lr_decay_factor ** (epoch // self.lr_decay_epoch))
        lr = max(lr, 1e-8)
        if epoch % self.lr_decay_epoch == 0:
            print ('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        print ('LR is set to {}'.format(lr))
        
        return optimizer

class CostSensitiveEngine:
    '''
    In CostSensitiveEngine, it is used for binary classification. 
    '''

    def __init__(self, model: nn.Module, loss_fn: nn.Module, device = "cpu", 
                 optimizer_fn= Adam, use_lr_scheduler = False, 
                 scheduler_lambda_fn = None, clip_gradient = 10000,  **optimizer_params):

        self.device = device
        self.model = model
        self.model.to(self.device)

        self.set_optimizer(optimizer_fn, **optimizer_params)
        self.use_lr_scheduler = use_lr_scheduler
        
        if self.use_lr_scheduler:
            self.scheduler = lr_scheduler.LambdaLR(self._optimizer, scheduler_lambda_fn) 

        self.loss_fn = loss_fn #Should get y_pred and y_true, becomes a method.
        self.loss_fn.to(self.device)
        self.epochs_trained = 0
        self.clip_gradient = clip_gradient

    def set_optimizer(self, optimizer_fn, **optimizer_params):
        '''
        Setting an optimizer, happens in the __init__
        '''
        
        self._optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)
        
    def forward(self, X, y):
        '''
        X is (N, M)
        y is (N,)

        The model should return predictions after normalization. (prediction between 0 and 1)
        '''
        y_pred = self.model(X).sigmoid() #Normalized
        loss = self.loss_fn(y_pred, y) #Loss is tensor
        y_arg_pred = (y_pred>0.5).to(torch.int16)


        cm = confusion_matrix(y, y_arg_pred)
        g_mean = np.sqrt(np.product(np.diag(cm) / cm.sum(axis=1)))
        accuracy = np.diag(cm).sum() / cm.sum()

        return {"loss": loss, 
                "accuracy": accuracy, 
                "g_mean": g_mean, 
                "mae": 1,
                "batch_size": X.shape[0],
                "confusion_matrix": cm}
        
    def _train_batch(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        
        self.model.train()
        self._optimizer.zero_grad()
        stats = self.forward(X, y)
        
        stats["loss"].backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient)
        

        self._optimizer.step()
        return stats

    def _train_epoch(self, loader):
        iterator = tqdm(loader, total = len(loader))
        
        cum_batch_size = 0 
        cum_loss = 0
        cum_accuracy = 0 
        cum_mae = 0

        for X, y in iterator: 
            stats = self._train_batch(X, y)
            n = stats["batch_size"]
            
            cum_batch_size += n
            cum_loss += stats["loss"].item() * n
            cum_accuracy += stats["accuracy"] * n #Total corrected so far
            cum_mae += stats["mae"] * n
            iterator.set_postfix(loss= cum_loss / cum_batch_size, 
                                 accuracy = cum_accuracy / cum_batch_size, 
                                 mae = cum_mae / cum_batch_size,
                                 batch_size = cum_batch_size)

        self.epochs_trained +=1
        
        if self.use_lr_scheduler:
            self.scheduler.step()

        return {
            "loss": cum_loss / cum_batch_size,
            "accuracy": cum_accuracy / cum_batch_size
            # "mae":cum_mae / cum_batch_size,
                }
    def _eval_batch(self, X, y):
        '''
        Evaluating the batch
        '''

        self.model.eval()
        X, y = X.to(self.device), y.to(self.device)
        with torch.no_grad():
            stats = self.forward(X, y)

        return stats
            
    def _eval_epoch(self, loader):
        '''
        Evaluating the entire epoch
        '''

        iterator = tqdm(loader, total = len(loader))
        
        cum_batch_size = 0 
        cum_loss = 0
        cum_accuracy = 0 
        cum_mae = 0
        cum_cm = np.array([[0 , 0],[0 , 0]],dtype=np.int32)

        for X, y in iterator:
            stats = self._eval_batch(X, y)
            n = stats["batch_size"]
            
            cum_batch_size += n
            cum_loss += stats["loss"].item() * n
            cum_accuracy += stats["accuracy"] * n #Total corrected so far
            cum_mae += stats["mae"] * n
            cum_cm += np.array(stats['confusion_matrix'],dtype=np.int32)
            iterator.set_postfix(loss= cum_loss / cum_batch_size, 
                                 accuracy = cum_accuracy / cum_batch_size, 
                                 mae = cum_mae / cum_batch_size,
                                 batch_size = cum_batch_size)

        return {
                "loss": cum_loss/cum_batch_size, 
                "accuracy":cum_accuracy/cum_batch_size,
                # "mae":cum_mae / cum_batch_size,
                "confusion_matrix" :cum_cm
               }
        
    def train(self, train_loader, test_loader=None, n_epochs=2):
        best_acc_train = 0
        best_acc_test = 0
        best_model = 0
        for _ in range(n_epochs):
            res = self._train_epoch(train_loader)
            if res['accuracy']> best_acc_train:
                best_acc_train = res['accuracy']

            if test_loader:
                res_test = self._eval_epoch(test_loader)
                if res_test['accuracy']> best_acc_test:
                    best_acc_test = res_test['accuracy']
                    best_model = copy.deepcopy(self.model)
        return best_acc_train,best_acc_test,best_model
