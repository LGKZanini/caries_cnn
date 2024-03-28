import torch # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
import torch.nn.functional as F # pyright: ignore[reportMissingImports]


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
            self.validation(val_data, epoch)
            
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
    
    def validation(self, val_data, epoch):
        
        y_val = []  # Lista para coletar y_val
        y_pred = []  # Lista para coletar y_pred
        loss_items = []  # Lista para coletar os itens de perda

        for X_test, y_test in val_data:
            
            X_test = X_test.to(f'cuda:{self.device}')
            y_test = y_test.to(f'cuda:{self.device}')
                                
            y_predicted = self.model(X_test)
            
            loss = self.loss_fn(y_predicted, y_test)
            loss_items.append(loss.item())  # Adicionando o item de perda à lista
            
            y_pred_model = F.softmax(y_predicted, dim=1).detach().cpu().numpy()
            y_pred.append(y_pred_model)  # Adicionando o array ao coletor de y_pred
            
            y_val.append(y_test.detach().cpu().numpy())  # Adicionando o array ao coletor de y_val

        # Convertendo listas para arrays do NumPy após a coleta
        y_val = np.concatenate(y_val, axis=0) if y_val else np.array([])
        y_pred = np.concatenate(y_pred, axis=0) if y_pred else np.array([])
        loss_item = np.array(loss_items) if loss_items else np.array([])

        self.get_metrics(y_val, y_pred, loss_item)
        
        return