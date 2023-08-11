from torch.utils.data import Dataset


class vcf_Dataset(Dataset):
    def __init__(self, ref, alt, tissue, label):
        self.ref, self.alt, self.tissue, self.label = ref, alt, tissue, label

    def __getitem__(self, index):
        ref = self.ref[index]
        alt = self.alt[index]
        tissue = self.tissue[index]
        label = self.label[index].float()
        return ref, alt, tissue, label

    def __len__(self):
        return len(self.label)


class gb_Dataset(Dataset):
    def __init__(self, seq, label):
        self.X, self.y = seq, label

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index].float()
        return X, y

    def __len__(self):
        return len(self.y)