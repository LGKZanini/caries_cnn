import gc
import torch
import numpy as np
import torchvision.transforms as T

from sklearn.metrics import confusion_matrix

class Trainer:

    def __init__(self, loss_fn, model, optimizer, scheduler, device=None):

        self.loss_fn = loss_fn
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_fn(self, X_train, y_train):

        y_predicted = self.model(X_train)

        loss = self.loss_fn(y_predicted, y_train)  
        loss.backward()
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item(), y_predicted

    def train_epochs(self, train_data, val_data, epochs=10):

        for epoch in range(epochs):
            
            print('Epoch', epoch+1)
            
            self.train_model(train_data)
            
            gc.collect()
                        
            self.validation(val_data)
            
            gc.collect()
            
            if epoch % 5 == 0 and epoch != 0:
                self.scheduler.step()
            
    def train_model(self, train_data):
        
        i = 0
        
        for X_train, y_train in train_data:
            
            X_train = X_train.to('cuda:'+str(self.device))
            y_train = y_train.to('cuda:'+str(self.device))
            
            loss, _ = self.train_fn(X_train, y_train)
            
            if i % 100 == 0:
                print(loss)
            
            i += 1
    
    def calculate_confusion_matrix(self, y_val, y_pred):

        return confusion_matrix(y_val, y_pred, labels=[0, 1, 2, 3, 4])
    
    def validation(self, val_data):
        
        y_val = []
        y_pred = []
        loss_item = []
        ok = False
        
        for X_test, y_test in val_data:
            
            X_test = X_test.to('cuda:'+str(self.device))
            y_test = y_test.to('cuda:'+str(self.device))
                             
            y_predicted = self.model(X_test)
            
            loss = self.loss_fn(y_predicted, y_test)
            loss_aux = loss.item()
            loss_item.append(loss_aux)
            
            y_pred_model = torch.sigmoid(y_predicted)
            
            if ok == False:
                
                y_pred = y_pred_model.detach().cpu().numpy().copy()
                y_val = y_test.detach().cpu().numpy().copy()
                    
                ok = True
                continue
            
            y_pred = np.append(y_pred, y_pred_model.detach().cpu().numpy(), axis=0)
            y_val = np.append(y_val, y_test.detach().cpu().numpy(), axis=0) 
            
        
        y_val = np.argmax(y_val, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        
        conf = self.calculate_confusion_matrix(y_val, y_pred)
        
        FP = conf.sum(axis=0) - np.diag(conf)  
        FN = conf.sum(axis=1) - np.diag(conf)
        TP = np.diag(conf)
        TN = conf.sum() - (FP + FN + TP)

        recall = TP/(TP+FN) 
        specificty = TN/(TN+FP) 
        precision = TP/(TP+FP) 
        acc = sum(np.diag(conf)) / conf.sum()

        print({
            "Loss_val" : sum(loss_item) / len(loss_item),
            "Accuracy" : acc,
            'Precision 0' : precision[0],
            'Precision 1' : precision[1],
            'Precision 2' : precision[2],
            'Precision 3' : precision[3],
            'Precision 4' : precision[4],
            'Specificity 0' : specificty[0],
            'Specificity 1' : specificty[1],
            'Specificity 2' : specificty[2],
            'Specificity 3' : specificty[3],
            'Specificity 4' : specificty[4],
            'Recall 0' : recall[0],
            'Recall 1' : recall[1],
            'Recall 2' : recall[2],
            'Recall 3' : recall[3],
            'Recall 4' : recall[4],
        })