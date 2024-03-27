import torch # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]

from sklearn.metrics import confusion_matrix # pyright: ignore[reportMissingImports]

class Trainer:

    def __init__(self, loss_fn, model, optimizer, scheduler, get_metrics, device=None, ssl=False):

        self.loss_fn = loss_fn
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.get_metrics = get_metrics

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
            self.validation(val_data)
            
            torch.cuda.empty_cache()
            
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
            
        return
    
    def validation(self, val_data):
        
        y_val = []
        y_pred = []
        loss_item = []
        
        for X_test, y_test in val_data:
            
            X_test = X_test.to('cuda:'+str(self.device))
            y_test = y_test.to('cuda:'+str(self.device))
                             
            y_predicted = self.model(X_test)
            
            loss = self.loss_fn(y_predicted, y_test)
            loss_item = np.concatenate(loss.item())
            
            y_pred_model = torch.sigmoid(y_predicted)
            
            y_pred = np.concatenate((y_pred, y_pred_model.detach().cpu().numpy()), axis=0)
            y_val = np.concatenate((y_val, y_test.detach().cpu().numpy()), axis=0)

        self.get_metrics(y_val, y_pred, loss_item)
        
        return