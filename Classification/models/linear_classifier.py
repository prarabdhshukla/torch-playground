import torch
import torch.nn as nn
import torch.optim as optim


class LogisitcRegression(nn.Module):

    def __init__(self, input_size, num_classes):
        super(LogisitcRegression, self).__init__()
        self.linear=nn.Linear(input_size,num_classes)
    
    def forward(self,x):
        out=self.linear(x)
        return out

    def train_model(self, train_loader, num_epochs=10, learning_rate=0.001):
        criterion=nn.CrossEntropyLoss()
        optimizer=optim.SGD(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                inputs=inputs.view(inputs.size(0),-1)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss=criterion(outputs,labels)
                loss.backward()
                optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
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