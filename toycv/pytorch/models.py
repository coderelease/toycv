import numpy
from torch.utils.data import DataLoader

numpy_dataset = numpy.random.randn(100, 3, 32, 32)

dataloader = DataLoader(numpy_dataset, batch_size=10, shuffle=True)


class EasyDataset:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
