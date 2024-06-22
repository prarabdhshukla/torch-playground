import models
import datasets
import torch
import argparse
import os
import inspect

def get_imported_classes(module):
    classes = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            classes.append(name)
    return classes

def parse_arguments():
    parser=argparse.ArgumentParser(description='Train the classifier')
    parser.add_argument('--model', help='Name of the model that needs to be trained. This should be the same name as the name of the class in models/', choices=get_imported_classes(models))
    parser.add_argument('--dataset', '-d', help='Name of the dataset that the classifier must be trained on. This should be the same name as the name of the corresponding class in datasets/', choices=get_imported_classes(datasets))

    return parser.parse_args()


if __name__=="__main__":
    args=parse_arguments()


    dataset_class=getattr(datasets, args.dataset)
    dataset=dataset_class()
    train_loader=dataset.load_data()

    input_size=3072
    num_classes=10

    model_class=getattr(models, args.model)
    model=model_class(input_size, num_classes)
    model.train_model(train_loader)

    predictions = model.predict(train_loader)
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(train_loader):
        total += labels.size(0)
        correct += (torch.tensor(predictions[i * train_loader.batch_size:(i + 1) * train_loader.batch_size]) == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')
