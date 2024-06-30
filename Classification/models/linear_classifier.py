import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class LogisticRegression(nn.Module):

    def __init__(self, input_size, num_classes, num_epochs, learning_rate, optimizer_class):
        super(LogisticRegression, self).__init__()
        self.linear=nn.Linear(input_size,num_classes)
        self.num_epochs=num_epochs
        self.lr=learning_rate
        self.optim_class=optimizer_class
    
    def forward(self,x):
        out=self.linear(x)
        return out

    def train_model(self, train_loader):
        criterion=nn.CrossEntropyLoss()
        optimizer=self.optim_class(self.parameters(), lr=self.lr)

        for epoch in tqdm(range(self.num_epochs), desc=f'Epoch: '):
            for inputs, labels in train_loader:
                inputs=inputs.view(inputs.size(0),-1)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss=criterion(outputs,labels)
                loss.backward()
                optimizer.step()
            
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')
    
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