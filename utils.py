import numpy as np

import torch
import random
import os
import pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from pathlib import Path
from sklearn.cluster import KMeans


def idx2bool(size, idx):
    v = np.zeros(size)
    for index in idx:
        v[index] = 1.0
    return v

def bool2idx(bool_array):
    return np.where(bool_array)[0]

def confirm_dir(path):
    path = Path(path)
    if not os.path.exists(path):
        path.mkdir(parents=True, exist_ok=True)


def update_args(cmdline_args, jsonFile):
    '''
    if there are confliction between jsonFile and cmdlines, cmdline_args overloads jsonFile. 
    '''
    class dotdict(dict):
        def __getattr__(self, name):
            return self[name]

        def __setattr__(self, name, value):
            self[name] = value
    args = dotdict()
    args.update(jsonFile)

    if "configs" in args:
        del args["configs"]
        jsonFile = jsonFile["configs"]
    

    for subp in [cmdline_args.dataset, remove_num_from_string(cmdline_args.net)]:
        jsonFile = jsonFile[subp]
        args.update(jsonFile)
        if "configs" in args:
            del args["configs"]
        if "configs" in jsonFile:
            jsonFile = jsonFile["configs"]

    args.update(cmdline_args.__dict__)

    for k, v in args.items():
        if k.split('_')[-1] =='dir':
            confirm_dir(v)

    args = update_img_transform(args)
    args = update_clip_rep_path(args)
    args = update_dynamics_path(args)
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.num_samples = args.num_org_samples + int(args.cor_rate * args.num_org_samples)

    
    return args

def remove_num_from_string(s):
    return ''.join([i for i in s if not i.isdigit()])


def update_img_transform(args):
    
    t = args.transforms['train']
    c = args.transforms['cor']
    
    args.train_transform = t.split('_')[-1]
    args.cor_transform = c.split('_')[-1]
    
    return args

def update_clip_rep_path(args):
    args.clip_rep_path = None
    if args.net =='clip':
        dir_path = os.path.join(args.save_dir, args.dataset, 'clip')
        confirm_dir(dir_path)
        args.clip_rep_path = os.path.join(dir_path, f'{args.train_transform}_{args.cor_transform}.pth' )
    return args


def update_dynamics_path(args, prefix=''):

    
    dynamics_dir = os.path.join(args.save_dir, args.dataset, 'dynamics')
    confirm_dir(dynamics_dir)

    if args.cor_rate == 0:
        cor_str = 'cor00' + args.train_transform
    else:

        if 'resnet' in args.net: 
            cor_str = f'cor{args.cor_rate*100:.0f}' + args.train_transform + 'crop' + str(args.resent_crop_size)
        else:
            cor_str = f'cor{args.cor_rate*100:.0f}' + args.train_transform + args.cor_transform

    noise_str = _set_noise_str(args.noise_type, args.noise_rate)



    file_name = f'{prefix}{args.net}_{noise_str}_{cor_str}_{args.lr}{args.lr_scheduler}_{args.num_epochs}ep.p'

    args.dynamics_path = os.path.join(dynamics_dir, file_name)
    print("Dynamic_path:", args.dynamics_path)
    return args


def _set_noise_str(noise_type, noise_rate):
    
    if noise_rate:
        noise_type_ = noise_type + '_' + f'{int(noise_rate * 100)}'
    else:
        noise_type_ = noise_type


    if noise_type_ in ['clean', 'human_aggre', 'human_worst', 'human_one', 'human_two', 'human_three','human_noisy100']:
            
        noise_str = noise_type
    
    else:
        noise_str = noise_type + f'{int(noise_rate * 100)}'
    
    return noise_str
    


def record_dynamics(index, logits, targets, recorder, n_class):
    '''
    output: n_data x epochs x 2 
    '''
    for i, ind in enumerate(index):
        logit = logits[i]
        target_label = targets[i].data
        target_logit = logit[target_label]
        
        if target_label < n_class -1:
            nontarget_logits = torch.cat([logit[:target_label], logit[target_label+1:]], dim=0)
        else:
            nontarget_logits = logit[:target_label]
        largest_other_logit, _  = nontarget_logits.max(dim=0)

        recorder[ind].append([target_logit.item(), largest_other_logit.item()])



def save_dynamics(recorder, loader, dynamics_path, saving_dynamics):
    
    '''
    saved data
    idx, record(N x epochs x 2), given_label, clean_label

    '''
    n_data = len(recorder)
    recorder = np.array(recorder).reshape((n_data, -1)) # n_data x (num_epochs x 2)

    if loader.dataset.is_cor:
        cor_idx = loader.dataset.cor_idx
        all_idx = list(range(loader.dataset.num_data)) + cor_idx.tolist()
    else:
        all_idx = list(range(loader.dataset.num_data))

    noisy_label = loader.dataset.train_noisy_label
    clean_label = loader.dataset.train_label

    record_all = np.column_stack((all_idx, recorder, noisy_label, clean_label))
    if saving_dynamics:
        with open(dynamics_path, "wb") as f1:
            pickle.dump(record_all, f1)
    return record_all



def load_dynamics(record_all):
    
    all_idx = record_all[:, 0].astype(np.int64)
    logits = record_all[:, 1:-2] # N x (epochs * 2)
    noise_label_arr = record_all[:, -2].astype(np.int64)  # array
    clean_label = record_all[:, -1].astype(np.int64)  # array
    num_data = all_idx.shape[0]

    logits = logits.reshape(num_data, -1, 2)  # N x epochs x 2
    logits = logits.transpose((0,2,1)) # N x 2 x epochs

    logits = torch.from_numpy(logits).cpu().float() # N x 2 x epochs
    
    logits = logits[:, 0,: ] - logits[:, 1,: ] # N x epochs 
    logits = torch.unsqueeze(logits, dim=1) # N x 1 x epochs
    
        

    return logits, clean_label.tolist(), noise_label_arr.tolist(), all_idx




def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def kmeans_centers(loader, net, device):
    input_rep = []
    embeddings = []
    net.eval()
    with torch.no_grad():
        for data in loader:
            inputs, _, _ = data
            input_rep.append(inputs)
            inputs = inputs.to(device)
            embeddings.append(net(inputs))
    input_rep = torch.cat(input_rep, dim=0).numpy()
    embeddings = torch.cat(embeddings, dim=0).cpu().numpy()
    
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(embeddings)
    c = torch.from_numpy(kmeans.cluster_centers_).to(device).to(inputs.dtype)

    idx_0 = bool2idx(kmeans.labels_ == 0)
    idx_1 = bool2idx(kmeans.labels_ == 1)
    avg_logit_0 =  np.mean(input_rep[idx_0][:,0,:]) # avg. logit for a given label
    avg_logit_1 = np.mean(input_rep[idx_1][:,0,:]) # avg. logit for a given label

    if avg_logit_0 > avg_logit_1:  # noisy label center having small logit in average 
        c_noisy = c[1, :]
        c_clean = c[0, :]
    else:
        c_noisy = c[0, :]
        c_clean = c[1, :]  
    


    c_noisy = torch.unsqueeze(c_noisy, dim=0)
    c_clean = torch.unsqueeze(c_clean, dim=0)
    return c_noisy, c_clean 


def average_center(loader, net, rep_dim, device):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    n_samples = 0
    c = torch.zeros(rep_dim, device=device)

    net.eval()
    with torch.no_grad():
        for data in loader:
            # get the inputs of the batch
            inputs, _, _,_ = data
            inputs = inputs.to(device)
            outputs = net(inputs)
            # outputs = F.normalize(outputs)
            n_samples += outputs.shape[0]
            c += torch.sum(outputs, dim=0)
    c /= n_samples



    c = torch.unsqueeze(c, dim=0)
    return c


def init_centroid(args, net, cor_loader, org_loader):


    if args.cor_rate==0:
        c_noisy, c_clean = kmeans_centers(org_loader, net, args.device, add_eps=args.add_eps_n, eps=args.eps)

    else:
        c_noisy = average_center(cor_loader, net, args.rep_dim, args.device)
        c_clean = average_center(org_loader, net, args.rep_dim, args.device)



    return c_noisy, c_clean 



def get_loader(dataset, mode='all', shuffle=False, batch_size=1024, num_workers=4, drop_last=False):

    if mode == 'original':
        idx = bool2idx(np.array(dataset.cor_label) == 0)
        dataset = Subset(dataset, idx)
    elif mode == 'corrupt':
        idx = bool2idx(np.array(dataset.cor_label) == 1)
        dataset = Subset(dataset, idx)

    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, drop_last=drop_last)

    return loader
    


def get_label(args, class_label_clean, class_label_noise):
    num_train = len(class_label_clean)
    noise_or_not = np.array(class_label_clean, dtype=object) != np.array(class_label_noise, dtype=object)
    cor_label_all = np.array(([0] * args.num_org_samples + [1] * (num_train - args.num_org_samples)))
    
    cor_label_sel = cor_label_all.astype(int)
    noise_label_sel= noise_or_not.astype(int)
    
    num_original = (cor_label_sel==0).sum()
    num_cor = (cor_label_sel==1).sum()
    assert num_train == num_cor + num_original

    overall_noise_rate = noise_label_sel.mean()
    original_noise_rate = noise_label_sel[:num_original].mean()
    
    if num_original == num_train:
        cor_noise_rate = 0.0
    else:
        cor_noise_rate = noise_label_sel[num_original:].mean()
    
    print(f'Noise_rate, # overall:{overall_noise_rate:.1%}, # original:{original_noise_rate:.1%}, # cor: {cor_noise_rate:.1%}')
    print(f'Num of train (orignal + cor): {num_train}, num_original:{num_original}, num_cor: {num_cor}')

    return cor_label_sel, noise_label_sel



class lable_inform:
    def __init__(self, cor_label, noise_label, all_idx):

        n_data = all_idx.shape[0]
        self.noise_bool = noise_label == 1 
        self.cor_bool = cor_label == 1 
        self.n_cor = self.cor_bool.sum()
        self.n_org = n_data - self.n_cor

        self.cor = bool2idx(self.cor_bool) # 1' 2' 3'
        self.corN = bool2idx(self.noise_bool * self.cor_bool) # 1' 2'
        org_pair_corN = all_idx[self.corN] # 2, 3
        orgC_pair_corN = bool2idx(idx2bool(n_data, org_pair_corN) * ~self.noise_bool) # 1
        orgN_pair_corN = bool2idx(idx2bool(n_data, org_pair_corN) * self.noise_bool) # 2
        
        corN_from_N = self.corN[self.noise_bool[org_pair_corN]] # 2'
        corN_from_C = np.setdiff1d(self.corN, corN_from_N) # 1'

        self.corC = bool2idx(~self.noise_bool * self.cor_bool) # 3'
        orgN_pair_orC = all_idx[self.corC] # 3

        self.org_pair_cor = all_idx[self.cor] # 1, 2, 3
        self.all_idx = all_idx
        
        #### original data
        self.org = bool2idx(~self.cor_bool)
        self.orgC = bool2idx(~self.noise_bool * ~self.cor_bool)
        self.orgN = bool2idx(self.noise_bool * ~self.cor_bool)




def get_idx_of_idx(org_idx, idx, device):
    '''
    org_dix, idx: numpy 
    '''
    return torch.cat([(org_idx == i).nonzero() for i in idx]).squeeze(dim=1).to(device)


