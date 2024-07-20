import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class LogisticRegression(nn.Module):

    def __init__(self, input_size, num_classes, num_epochs, learning_rate, optimizer_class, metric, est):
        super(LogisticRegression, self).__init__()
        self.linear=nn.Linear(input_size,num_classes)
        self.num_epochs=num_epochs
        self.lr=learning_rate
        self.optim_class=optimizer_class
        self.metric = metric
        self.est = est
    
    def forward(self,x):
        out=self.linear(x)
        return out

    def train_model(self, train_loader, val_loader=None):
        criterion=nn.CrossEntropyLoss()
        optimizer=self.optim_class(self.parameters(), lr=self.lr)
        val_loss = torch.Tensor([0])
        loss = torch.Tensor([0])
        val_metric = 'NA'
        es_counter, best_val_loss = 0,0
        pbar = tqdm(range(self.num_epochs), desc=f'Train Loss: {loss.item():.4f} Val Loss: {val_loss.item():0.4f} Val {self.metric.__name__}: {val_metric}')
        for epoch in pbar:
            for inputs, labels in train_loader:
                inputs=inputs.view(inputs.size(0),-1)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss=criterion(outputs,labels)
                loss.backward()
                optimizer.step()
            
            if val_loader:
                self.eval()
                val_loss = 0
                outputs = []
                ground_truth = []
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs=inputs.view(inputs.size(0),-1)
                        output = self(inputs)
                        outputs.append(output)
                        ground_truth.append(labels)
                        val_loss += criterion(output, labels)

                val_loss/= len(val_loader.dataset)  
                val_metric = self.metric(outputs, ground_truth)

            description=f'Train Loss: {loss.item()/len(train_loader.dataset):.4f} Val Loss: {val_loss.item():0.4f} Val {self.metric.__name__}: {val_metric}'
            pbar.set_description(description)
            self.train()

            if self.est:
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()
                    es_counter = 0
                else:
                    es_counter+=1
                    if es_counter > self.est:
                        print(f"\nEarly Stopping... Epochs: {epoch}/{self.num_epochs} \n Train Loss: {loss.item()/len(train_loader.dataset):.4f} Val Loss: {val_loss.item():0.4f} Val {self.metric.__name__}: {val_metric}")
                        return val_metric
        return val_metric

            # print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
    
    def predict(self, test_loader):
        self.eval()
        predictions=[]
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs=inputs.view(inputs.size(0),-1)
                outputs=self(inputs)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.tolist())
        return predictions