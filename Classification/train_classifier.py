import models
import datasets
import metrics
import torch
import argparse
import os
import inspect
import torch.optim as optim

def get_imported_classes(module):
    classes = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            classes.append(name)
    return classes

def get_imported_functions(module):
    functions = []
    for name, value in vars(module).items():
        if callable(value):
            functions.append(name)
    return functions

def parse_arguments():
    parser=argparse.ArgumentParser(description='Train the classifier')
    parser.add_argument('--model', help='Name of the model that needs to be trained. This should be the same name as the name of the class in models/', choices=get_imported_classes(models))
    parser.add_argument('--dataset', '-d', help='Name of the dataset that the classifier must be trained on. This should be the same name as the name of the corresponding class in datasets/', choices=get_imported_classes(datasets))
    parser.add_argument('--optimizer', '-o', help='The optimizer to be used. The optimizer class must belong to torch.optim')
    parser.add_argument('--learning_rate', '-lr', help='Learning rate', default=0.001, type=float)
    parser.add_argument('--num_epochs', '-ne', help='Number of epochs for training', default=10, type=int)
    parser.add_argument('--save_dir', help='Directory where the trained model must be saved', default='./saved_models/')
    parser.add_argument('--save_as', help='File name of the saved model', default=None)
    parser.add_argument('--metric', help='Validation Metric', default='accuracy', choices=get_imported_functions(metrics))
    parser.add_argument('--early_stopping_threshold','-est', help='Number of epochs to wait after validation loss does not improve. Early stopping is not implemented if argument not provided.', default=None, type=int)

    return parser.parse_args()


if __name__=="__main__":
    args=parse_arguments()


    dataset_class=getattr(datasets, args.dataset)
    dataset=dataset_class()
    train_loader,val_loader=dataset.load_data()

    input_size=dataset.input_size
    num_classes=dataset.num_classes

    optimizer_class= getattr(optim, args.optimizer)

    metric = getattr(metrics, args.metric)

    model_class=getattr(models, args.model)
    model=model_class(input_size, num_classes, args.num_epochs, args.learning_rate, optimizer_class, metric, args.early_stopping_threshold)

    model.train_model(train_loader, val_loader)

    if args.save_as is not None:
        save_dest=os.path.join(args.save_dir, f'{args.save_as}.pth')
        torch.save(model.state_dict(), save_dest)

    
