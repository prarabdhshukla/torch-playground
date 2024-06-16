from models.linear_classifier import LogisitcRegression
from Classification.datasets import Cifar10
import torch


if __name__=="__main__":
    dataset=Cifar10()
    train_loader=dataset.load_data()

    input_size=3072
    num_classes=10

    model=LogisitcRegression(input_size, num_classes)
    model.train_model(train_loader)

    predictions = model.predict(train_loader)
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(train_loader):
        total += labels.size(0)
        correct += (torch.tensor(predictions[i * train_loader.batch_size:(i + 1) * train_loader.batch_size]) == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
