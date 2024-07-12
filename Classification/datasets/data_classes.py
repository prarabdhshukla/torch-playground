from torchvision.datasets import CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

Transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

class Cifar10:
    def __init__(self, root='./data', train= True, download=True, transform = Transform, val_size=0.2):
        self.root=root
        self.train=train
        self.download=download
        self.num_classes=10
        self.input_size=3072
        self.transform = transform
        self.val_size=val_size
    
    def load_data(self, batch_size=32, shuffle=True, num_workers=2):

        if self.train:
            self.data_obj = CIFAR10(root=self.root, train=self.train, download=self.download, transform=self.transform)
            self.train_obj, self.val_obj = random_split(self.data_obj, [1-self.val_size, self.val_size])

            return DataLoader(self.train_obj,batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), DataLoader(self.val_obj, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        else:
            self.test_data = CIFAR10(root=self.root, train=self.train, download = self.download, transform= self.transform )
            return DataLoader(self.test_data,batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


class Cifar100:
    def __init__(self, root='./data', train= True, download=True, transform = Transform, val_size=0.2):
        self.root=root
        self.train=train
        self.download=download
        self.num_classes=100
        self.input_size=3072
        self.transform = transform
        self.val_size=val_size
    
    def load_data(self, batch_size=32, shuffle=True, num_workers=2):

        if self.train:
            self.data_obj = CIFAR100(root=self.root, train=self.train, download=self.download, transform=self.transform)
            self.train_obj, self.val_obj = random_split(self.data_obj, [1-self.val_size, self.val_size])

            return DataLoader(self.train_obj,batch_size=batch_size, shuffle=shuffle, num_workers=num_workers), DataLoader(self.val_obj, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        else:
            self.test_data = CIFAR100(root=self.root, train=self.train, download = self.download, transform= self.transform )
            return DataLoader(self.test_data,batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
