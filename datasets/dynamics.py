

from torch.utils.data import Dataset

class cluster_dataset(Dataset):
    '''
    cor_label added
    '''
    def __init__(self, data, noise_label, cor_label):
        self.data = data
        self.noise_label = noise_label
        self.cor_label = cor_label

    def __getitem__(self, index):
        
        
        return self.data[index], self.noise_label[index], self.cor_label[index], index  
    
    def __len__(self):
        return self.data.shape[0]

