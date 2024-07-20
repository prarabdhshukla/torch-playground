import models
import Datasets
import metrics
import torch
import argparse
import os
import inspect
import json
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
    parser.add_argument('--dataset', '-d', help='Name of the dataset that the classifier must be trained on. This should be the same name as the name of the corresponding class in datasets/', choices=get_imported_classes(Datasets))
    parser.add_argument('--optimizer', '-o', help='The optimizer to be used. The optimizer class must belong to torch.optim')
    parser.add_argument('--learning_rate', '-lr', help='Learning rate', default=0.001, type=float)
    parser.add_argument('--num_epochs', '-ne', help='Number of epochs for training', default=10, type=int)
    parser.add_argument('--batch_size', '-b', help='Batch size for training', default = 32, type = int )
    parser.add_argument('--val_size', help='Fraction of training set to be used as validation set', type= float, default = 0.2)
    parser.add_argument('--save_dir', help='Directory where training run data (including models) will be saved.', default='./saved_models/')
    parser.add_argument('--save_as', help='The common base name for the saved model state dictionary and training metadata JSON file. \'metadata\' will be appended to the file name for the metadata JSON file. Pass None if you want to waste compute and do not want to save the model.', default=None)
    parser.add_argument('--save_folder', help='The name of the subfolder within the --save_dir where the model state dictionary, the training metadata JSON file and the training log will be saved. If not provided, the name specified by --save_as is used.', default=None)
    parser.add_argument('--skip_metadata', help="Do not save the training metadata (hyperparameters) to a JSON file. Use this if you\'re feeling rebellious—or just like living on the edge!", action = 'store_true')
    parser.add_argument('--metric', help='Validation Metric', default='accuracy', choices=get_imported_functions(metrics))
    parser.add_argument('--early_stopping_threshold','-est', help='Number of epochs to wait after validation loss does not improve. Early stopping is not implemented if argument not provided.', default=None, type=int)
    parser.add_argument('--log_training', '-l', action='store_true', help="Enable logging. If specified, a training log will be saved to the file path constructed from the `--save_dir`, `--save_folder`, and `--save_as` arguments, resulting in <save_dir>/<save_folder>/<save_as>_training_log.json")


    return parser.parse_args()


if __name__=="__main__":
    args=parse_arguments()


    dataset_class=getattr(Datasets, args.dataset)
    dataset=dataset_class(val_size = args.val_size)
    train_loader,val_loader=dataset.load_data(batch_size=args.batch_size)

    input_size=dataset.input_size
    num_classes=dataset.num_classes

    optimizer_class= getattr(optim, args.optimizer)

    metric = getattr(metrics, args.metric)

    model_class=getattr(models, args.model)
    model=model_class(input_size, num_classes, args.num_epochs, args.learning_rate, optimizer_class, metric, args.early_stopping_threshold, args.log_training)

    val_metric = model.train_model(train_loader, val_loader)

    if args.save_folder:
        save_folder = args.save_folder
    else:
        save_folder = args.save_as
    
    if args.save_as is not None:
        os.makedirs(os.path.join(args.save_dir, save_folder), exist_ok=True)
        save_dest=os.path.join(args.save_dir, save_folder, f'{args.save_as}.pth')
        torch.save(model.state_dict(), save_dest)

        if not args.skip_metadata:
            meta_data = { 'Dataset': args.dataset,
                          'Validation_size': args.val_size,
                          'Input_size': input_size,
                          'Num_classes': num_classes,
                          'Batch_size' : args.batch_size,
                          'Optimizer' : args.optimizer,
                          'num_epochs': args.num_epochs,
                          'learning_rate' : args.learning_rate,
                          'Early_stopping_threshold' : args.early_stopping_threshold,
                          'validation_metric' : args.metric,
                          'Validation_metric_value': val_metric 
                         }
            with open(os.path.join(args.save_dir, save_folder, f'{args.save_as}_metadata.json'), 'w') as f:
                json.dump(meta_data, f, indent= 4)
        
        if args.log_training:
            with open(os.path.join(args.save_dir, save_folder, f'{args.save_as}_training_log.json'), 'w') as f:
                json.dump(model.log, f, indent=4)


    
