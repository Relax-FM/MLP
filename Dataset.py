import torch
from torch.utils.data import Dataset

class NASDAQDataSet(Dataset):
    def __init__(self, data=None):
        if (data==None):
            raise Exception("Plz, entry all data")
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        info = torch.tensor(self.data[index][0])
        label = torch.tensor(self.data[index][1])
        info = info.to(torch.float32)
        label = label.to(torch.float32)
        return (info, label)
