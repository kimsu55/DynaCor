from __future__ import print_function

import os
import os.path
import sys
import json

import numpy as np
import pandas as pd
from PIL import Image
import _pickle as cPickle


import pickle
import torch
from torch.utils.data import Dataset
from utils import bool2idx
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, Lambda, RandomHorizontalFlip, RandomCrop

import numpy as np
import random


from .utils_data import download_url, check_integrity, noisify_instance, multiclass_noisify, noisify ,get_random_idx


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def unpickle(file):
    with open(file, "rb") as fo:
        return cPickle.load(fo, encoding="latin1")


def _convert_RGB(image):
    return image.convert("RGB")


transform_10_none = Compose(
    [
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_10_weak = Compose(
    [
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_100_none = Compose(
    [
        ToTensor(),
        Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ]
)

transform_100_weak = Compose(
    [
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ]
)

transform_clip_none= Compose(
    [   
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Lambda(_convert_RGB),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]
)

transform_clip_flip= Compose(
    [   
        Resize(224, interpolation=BICUBIC),
        RandomHorizontalFlip(p=1),
        CenterCrop(224),
        Lambda(_convert_RGB),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]
)



class cifar_dataset_noisy(Dataset):
    '''
    CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    '''

    def __init__(
        self,
        dataset,
        root_dir,
        mode,
        transform=None,
        noise_type="clean", 
        noise_rate=0, 
        noise_file="",
        cor_rate=0, 
        random_state=0,  **kwargs
    ):
        self.dataset = dataset
        self.root_dir = root_dir
        self.run_mode = mode
        self.cor_rate = cor_rate
        self.is_cor = False
        
        self.noise_type = noise_type
        self.noise_rate = noise_rate
        self.noise_file = noise_file
        self.random_state = random_state
             
        if transform:
            self.transform = {}
            for key, v in transform.items():
                self.transform[key] = globals()[v]  

        self.__dict__.update(kwargs)

        self.download(dataset)
        self.basic_process()

        if 'clean' not in self.noise_type:
            self.noisy_process()

        if (self.run_mode != "test") and (self.cor_rate > 0):
            self.is_cor = True
            self.corrupting_process()


    def basic_process(self):    
        if self.run_mode == "test":
            if self.dataset == "cifar10":
                file = os.path.join(self.root_dir, self.base_folder, "test_batch")   
                test_dic = unpickle(file)
                self.test_data = test_dic["data"]
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic["labels"]
            elif self.dataset == "cifar100":
                file = os.path.join(self.root_dir, self.base_folder, "test")  
                # test_dic = unpickle("%s/test" % self.root_dir)
                test_dic = unpickle(file)
                self.test_data = test_dic["data"]
                self.test_data = self.test_data.reshape((10000, 3, 32, 32))
                self.test_data = self.test_data.transpose((0, 2, 3, 1))
                self.test_label = test_dic["fine_labels"]
            self.num_data = self.test_data.shape[0]
        else:
            train_data = []
            train_label = []
            if self.dataset == "cifar10":
                for n in range(1, 6):
                    file = os.path.join(self.root_dir, self.base_folder, f"data_batch_{n}")   
                    data_dic = unpickle(file)
                    train_data.append(data_dic["data"])
                    train_label = train_label + data_dic["labels"]
                train_data = np.concatenate(train_data)
            elif self.dataset == "cifar100":
                file = os.path.join(self.root_dir, self.base_folder, "train")   
                train_dic = unpickle(file)
                train_data = train_dic["data"]
                train_label = train_dic["fine_labels"]
            train_data = train_data.reshape((50000, 3, 32, 32))
            train_data = train_data.transpose((0, 2, 3, 1))

            self.train_label = train_label # list
            self.train_data = train_data
            self.num_data = train_data.shape[0]


    def noisy_process(self):
            '''
            train_labels_arr: N arrray  
            self.train_noisy_labels: list
            self.train_labels: list

            '''
            if self.noise_type in ['sym', 'asym']:
                
                self.train_noisy_label_arr, actual_noise_rate = noisify(self.nb_classes, self.train_label, self.noise_type, self.noise_rate, self.random_state)
                self.train_noisy_label = self.train_noisy_label_arr.tolist()

            elif self.noise_type == 'instance':
                self.train_noisy_label, actual_noise_rate = noisify_instance(self.train_data, self.train_label, noise_rate=self.noise_rate)
                self.train_noisy_label_arr = np.array(self.train_noisy_label)
                                                                         
            else:  # CIFAR-10/100N human label 
                
                _type = self.noise_type.split('_')[1]
                print(f'noisy label loaded from {self.noise_file}')
                self.train_noisy_label_arr = self.load_label_human_cifarN(_type)
                
                if 'Syn' in self.noise_type:
                    T = np.zeros((self.nb_classes, self.nb_classes))
                    for i in range(self.train_noisy_label_arr.shape[0]):
                        T[self.train_label[i]][self.train_noisy_label_arr[i]] += 1
                    T = T / np.sum(T, axis=1)
                    print(f'Noise transition matrix is \n{T}')
                    self.train_noisy_label_arr = multiclass_noisify(np.expand_dims(np.array(self.train_label), axis=1), P=T, random_state=self.random_state)  # np.random.randint(1,10086)
                    self.train_noisy_label_arr = np.squeeze(self.train_noisy_label_arr, axis=1)
                    
                    T = np.zeros((self.nb_classes, self.nb_classes))
                    for i in range(self.train_noisy_label_arr.shape[0]):
                        T[self.train_label[i]][self.train_noisy_label_arr[i]] += 1
                    T = T / np.sum(T, axis=1)
                    print(f'New synthetic noise transition matrix is \n{T}')
                
                self.train_noisy_label = self.train_noisy_label_arr.tolist()
                

            self.noise_or_not = np.transpose(self.train_noisy_label) != np.transpose(self.train_label)
            actual_noise_rate = self.noise_or_not.sum() / self.noise_or_not.shape[0]
            
            print(f'noise_type: {self.noise_type}, noise_ratio(setting):{self.noise_rate:.2%}, actual noise_ratio: {actual_noise_rate:.2%}')
            self.noise_rate =  actual_noise_rate


    def corrupting_process(self):

        self.cor_idx = self._get_cor_index()

        cor_data = self.train_data[self.cor_idx]
        cor_label_arr = np.array(self.train_label)[self.cor_idx] # original clean label
        cor_noise_label_arr = self.train_noisy_label_arr[self.cor_idx] # original noisy label
        
        if self.keep_org_label:
            self._combine_data(cor_data, cor_label_arr, cor_noise_label_arr)
        elif self.cor_randomflip:
            cor_noise_label_arr_corrupted, _ = noisify(self.nb_classes, cor_noise_label_arr.tolist() , "randomflip", 1.0)
            self._combine_data(cor_data, cor_label_arr, cor_noise_label_arr_corrupted)
        else:
            cor_noise_label_arr_corrupted, _ = noisify(self.nb_classes, cor_noise_label_arr.tolist() ,"sym" , 1.0)
            self._combine_data(cor_data, cor_label_arr, cor_noise_label_arr_corrupted)

    
    def _combine_data(self, cor_data, cor_label_arr, cor_noisy_label_arr_corrupted):
        self.train_data = np.concatenate([self.train_data, cor_data], axis=0)
        self.train_label = self.train_label + cor_label_arr.tolist()
        self.train_noisy_label = self.train_noisy_label + cor_noisy_label_arr_corrupted.tolist()
        
        cor_noise_or_not = cor_label_arr != cor_noisy_label_arr_corrupted  # N x 1
        self.noise_or_not = np.concatenate([self.noise_or_not, cor_noise_or_not])

        cor_noise_rate = cor_noise_or_not.sum() / cor_data.shape[0]
        revised_noise_rate = self.noise_or_not.sum()/ self.noise_or_not.shape[0]
        print(f"corrupting ratio: {self.cor_rate:.0%}, noise rate of corrupted data:{cor_noise_rate:.2%}, overall noise_rate: {revised_noise_rate:.2%}%") 


    def _get_cor_index(self):
        test_idx  = get_random_idx(self.num_data, 0.1, self.random_state)
        all_idx = list(range(self.num_data))
        train_val_idx = np.setdiff1d(all_idx, test_idx) # ordered 
        num_cor_data = int(self.num_data * self.cor_rate)

        if num_cor_data > train_val_idx.shape[0]:
            num_cor_data = train_val_idx.shape[0]

        _cor_ratio = num_cor_data/train_val_idx.shape[0]
        cor_idx = get_random_idx(train_val_idx, _cor_ratio, self.random_state)
        
        return cor_idx
        

    def __getitem__(self, index):

        if self.run_mode == "train":
            img, target = self.train_data[index], self.train_noisy_label[index]
            img = Image.fromarray(img)
            
            if index > self.num_data:
                img = self.transform["cor"](img)
            else:
                img = self.transform["train"](img)
        
            return img, target, index

        elif self.run_mode == "test":
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform["test"](img)
            return img, target


    def __len__(self):
        if self.run_mode != "test":
            return len(self.train_data)
        else:
            return len(self.test_data)


    def _check_integrity(self, train_list, test_list):
        root = self.root_dir
        for fentry in (train_list + test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


    def download(self, dataset):
        import tarfile
        
        root = self.root_dir
        if dataset=='cifar10':
            self.base_folder = 'cifar-10-batches-py'
            self.nb_classes = 10
            url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
            filename = "cifar-10-python.tar.gz"
            tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
            train_list = [
                ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
                ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
                ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
                ['data_batch_4', '634d18415352ddfa80567beed471001a'],
                ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
            ]

            test_list = [
                ['test_batch', '40351d587109b95175f43aff81a1287e'],
            ]
        elif dataset=='cifar100':
            self.base_folder = 'cifar-100-python'
            self.nb_classes = 100
            url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
            filename = "cifar-100-python.tar.gz"
            tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
            train_list = [
                ['train', '16019d7e3df5f24257cddd939b257f8d'],
            ]
            test_list = [
                ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
            ]
        else:
            raise RuntimeError('Dataset not found')

        if self._check_integrity(train_list, test_list):
            print('Files already downloaded and verified')
            return

        download_url(url, root, filename, tgz_md5)
        
        # extract file
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, filename), "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)

    
    def load_label_human_cifarN(self, noise_type): 
        # NOTE only load manual training label
        noise_label = torch.load(self.noise_file)
        noise_type_map = {'worst': 'worse_label', 'aggre': 'aggre_label', 'one': 'random_label1', 'two': 'random_label2', 'three': 'random_label3', 'noisy100': 'noisy_label'}  ## CIFAR-10/100N keys
        noise_type = noise_type_map[noise_type]
        if isinstance(noise_label, dict):
            if "clean_label" in noise_label.keys():
                clean_label = torch.tensor(noise_label['clean_label'])
                assert torch.sum(torch.tensor(self.train_label) - clean_label) == 0
                print(f'Loaded {noise_type} from {self.noise_file}.')
                print(f'The overall noise rate is {1 - np.mean(clean_label.numpy() == noise_label[noise_type])}')
            return noise_label[noise_type].reshape(-1)
        else:
            raise Exception('Input Error')


    def get_statistics(self):

        actual_noise_rate =  self.noise_or_not.sum() / self.noise_or_not.shape[0]
        num_noise = self.noise_or_not.sum()
        hard_or_not = self.c_score < 0.5
        num_hard = hard_or_not.sum()
        clean_easy = (~self.noise_or_not * ~hard_or_not).sum()
        clean_hard = (~self.noise_or_not * hard_or_not).sum()
        noise_easy = (self.noise_or_not * ~hard_or_not).sum()
        noise_hard = (self.noise_or_not * hard_or_not).sum()
        print(f'{self.noise_type}, noisy_rate(setting):{self.noise_rate:.1%}, (actual):{actual_noise_rate:.1%}')
        print(f'num_hard: {num_hard}, num_noisy:{num_noise}')
        print(f'clean_easy:{clean_easy}, clean_hard:{clean_hard}',)  
        print(f'noise_easy:{noise_easy}, noise_hard:{noise_hard}')


class cifar_dataset_noisy_clip(cifar_dataset_noisy):
    def __init__(
        self,
        dataset,
        root_dir,
        mode,
        transform=None,
        noise_type="clean", 
        noise_rate=0, 
        noise_file="",
        cor_rate=0, 
        random_state=0, clip_rep_path=''  
    ):
        super().__init__(
        dataset,
        root_dir,
        mode,
        transform,
        noise_type, 
        noise_rate, 
        noise_file,
        cor_rate, 
        random_state, 
        clip_rep_path=clip_rep_path)

    
    def basic_process(self):
        clip_reps = pickle.load(open(self.clip_rep_path, "rb"))
        if self.run_mode == "test":
            self.test_data = clip_reps['test'][0]
            self.test_label = clip_reps['test'][1].tolist()
            self.num_data = self.test_data.shape[0] 
        else:
            self.train_data = clip_reps['train'][0]
            self.train_label = clip_reps['train'][1].tolist()
            self.num_data = self.train_data.shape[0] 
    
    
    def corrupting_process(self):
        self.cor_idx = self._get_cor_index()
        
        clip_reps = pickle.load(open(self.clip_rep_path, "rb"))
        assert (self.train_label == clip_reps['cor'][1]).all()
        
        cor_data = clip_reps['cor'][0][self.cor_idx]
        cor_label_arr = clip_reps['cor'][1][self.cor_idx]  # original clean label
        cor_noise_label_arr = self.train_noisy_label_arr[self.cor_idx] # original noisy label


        cor_noise_label_arr_corrupted, _ = noisify(self.nb_classes, cor_noise_label_arr.tolist() , "sym", 1.0 )
        self._combine_data(cor_data, cor_label_arr, cor_noise_label_arr_corrupted)


    def __getitem__(self, index):

        if self.run_mode == "train":
            feature, target = self.train_data[index], self.train_noisy_label[index]
            return feature, target, index
        
        elif self.run_mode == "test":
            feature, target = self.test_data[index], self.test_label[index]
            return feature, target


class cifar_dataset_noisy_resnet(cifar_dataset_noisy):
    '''
    load transformed img and use it 
    '''
    def __init__(
        self,
        dataset,
        root_dir,
        mode,
        transform=None,
        noise_type="clean", 
        noise_rate=0, 
        noise_file="",
        cor_rate=0, 
        random_state=0, img_data_path=''
    ):
        super().__init__(
        dataset,
        root_dir,
        mode,
        transform,
        noise_type, 
        noise_rate, 
        noise_file,
        cor_rate, 
        random_state,
        img_data_path=img_data_path)

    def corrupting_process(self):
        modified_traindata = pickle.load(open(self.img_data_path, "rb"))

        self.cor_idx = self._get_cor_index()

        cor_data = modified_traindata[self.cor_idx]
        cor_label_arr = np.array(self.train_label)[self.cor_idx] # original clean label
        cor_noise_label_arr = self.train_noisy_label_arr[self.cor_idx] # original noisy label
        
        cor_noise_label_arr_corrupted, _ = noisify(self.nb_classes, cor_noise_label_arr.tolist() ,"sym" , 1.0)
        self._combine_data(cor_data, cor_label_arr, cor_noise_label_arr_corrupted)

    def __getitem__(self, index):

        if self.run_mode == "train":
            img, target = self.train_data[index], self.train_noisy_label[index]
            img = Image.fromarray(img)
            img = self.transform["train"](img)
        
            return img, target, index

        elif self.run_mode == "test":
            img, target = self.test_data[index], self.test_label[index]
            img = Image.fromarray(img)
            img = self.transform["test"](img)
            return img, target


