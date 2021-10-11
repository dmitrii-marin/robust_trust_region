import torch.utils.data.dataset

class IndexedDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, base):
        self.base = base

    def __getitem__(self, index):
        sample = self.base[index]
        sample["index"] = index
        return sample

    def __len__(self):
        return len(self.base)
