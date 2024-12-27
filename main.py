
import argparse
import random
import json
import torchvision

import numpy as np
import torch

from models import *
import torch.backends.cudnn as cudnn
import torch.optim as optim


from torch.utils.data import DataLoader
from datasets.dynamics import cluster_dataset
from torch.optim.lr_scheduler import  CosineAnnealingLR, MultiStepLR

from datasets.cifar import cifar_dataset_noisy, cifar_dataset_noisy_clip, cifar_dataset_noisy_resnet
from datasets.dynamics import  cluster_dataset
from utils import *
from sklearn.metrics import f1_score
import datasets.cifar as Data_cifar
import clip
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image, ImageOps



def load_args():

    parser = argparse.ArgumentParser(description='corrupted augmentation - record learning dynamics & fully supervised learning')
    
    parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, cifar100')
    parser.add_argument('--net', type=str, default='clip', help='resnet34, clip')
    parser.add_argument('--noise_type', type=str, default='asym', choices=['clean', 'sym', 'asym', 'instance', 'human_aggre', 'human_worst','human_one', 'human_two', 'human_three', 'human_noisy100']) # 


    parser.add_argument('--noise_rate', type=float, default=0.3)
    parser.add_argument('--cor_rate', type=float, default=0.10)
    parser.add_argument('--alpha', type=float, default=0.5)
    
    parser.add_argument('--cfg_file', type=str, default="./presets.json" )


    args = parser.parse_args()
    with open(args.cfg_file, "r") as f:
        jsonFile = json.load(f)

    args = update_args(args, jsonFile)

    return args



def get_features_labels(net, loader, use_cuda=True):

    net.eval()
    features_list, labels_list = [], []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            if use_cuda:
                images, labels = images.cuda(), labels.cuda()
            features = net.encode_image(images)
            features_list.append(features.detach().cpu().numpy())
            labels_list.append(labels.detach().cpu().numpy())
    features = np.squeeze(np.concatenate(features_list))
    labels = np.concatenate(labels_list)
    return features, labels



def generate_clip_rep(args, file_name):
    print("Generating image representation for clip model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load('ViT-B/32', device, jit=False)  # RN50, RN101, RN50x4, ViT-B/32
    
    saved = {}
    for k, v in args.transforms.items():
        transform = getattr(Data_cifar, v)

        if k == 'test':
            train_bool = False
        else:
            train_bool = True

        if args.dataset == 'cifar10':
            dataset = torchvision.datasets.CIFAR10(root=args.root_dir, train=train_bool, download=True, transform=transform)
        elif args.dataset == "cifar100":
            dataset = torchvision.datasets.CIFAR100(root=args.root_dir, train=train_bool, download=True, transform=transform)

        loader = DataLoader(dataset=dataset, batch_size=512, shuffle=False, num_workers=args.num_workers)
        features, labels = get_features_labels(model, loader)

        saved[k]=[features, labels]
    with open(file_name,"wb") as f1:
        pickle.dump(saved, f1) # {'orginal_train':[featues, labels], orginal_test':[featues, labels], 'cor': [features, labels]}
    print(f'Clip respresntions of original/corrupted {args.dataset} images are saved at: {file_name}')


def generate_resnet_rep_for_cor(args,  file_name, crop_size=4):
    print("Generating image representation for resnet")
    transform_10_none = Compose(
    [
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

    if args.dataset == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(root=args.root_dir, train=True, download=True, transform=transform_10_none)
    elif args.dataset == "cifar100":
        dataset = torchvision.datasets.CIFAR100(root=args.root_dir, train=True, download=True, transform=transform_100_none)

    
    
    data_arr = dataset.data #num_data x 32 x 32 x 3
    data_arr_t =[]
    for i in range(data_arr.shape[0]):
        img = Image.fromarray(data_arr[i])
        img_crop = ImageOps.crop(img, crop_size)
        img_resize = ImageOps.fit(img_crop, (32,32))
        data_arr_t.append(np.array(img_resize))

    data_arr_t = np.stack(data_arr_t)
    assert data_arr.shape == data_arr_t.shape
    assert data_arr.max() == data_arr_t.max()
    assert data_arr.min() == data_arr_t.min()

    with open(file_name,"wb") as f1:
        pickle.dump(data_arr_t, f1)


def build_classifier(args):
    if 'resnet' in args.net:
        model =  ResNet34(args.num_classes)   
        
        train_dataset = cifar_dataset_noisy_resnet(
                    dataset=args.dataset,
                    root_dir=args.root_dir,
                    mode='train',
                    transform=args.transforms,
                    noise_type=args.noise_type,
                    noise_rate=args.noise_rate,
                    noise_file=args.label_file_path,
                    cor_rate=args.cor_rate,
                    random_state=args.random_state,
                    img_data_path= args.modified_traindata_path)
        
        test_dataset = cifar_dataset_noisy(
                    dataset=args.dataset,
                    root_dir=args.root_dir,
                    mode='test',
                    transform=args.transforms,
                    )
        model = torch.nn.DataParallel(model).cuda()

     
    
    elif args.net == 'clip':  

        model = MLP([512, 512], args.num_classes)  # Lin-relu-Lin
  
        train_dataset = cifar_dataset_noisy_clip(
                        dataset=args.dataset,
                        root_dir=args.root_dir,
                        mode='train',
                        transform=args.transforms,
                        noise_type=args.noise_type,
                        noise_rate=args.noise_rate,
                        noise_file=args.label_file_path,
                        cor_rate=args.cor_rate,
                        random_state=args.random_state,
                        clip_rep_path=args.clip_rep_path)
        test_dataset = cifar_dataset_noisy_clip(
                        dataset=args.dataset,
                        root_dir=args.root_dir,
                        mode='test',
                        transform=args.transforms,
                        clip_rep_path=args.clip_rep_path)
        model = model.cuda()
        

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
            )
    test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
            )
    if args.lr_scheduler == 'multi':
        assert args.num_epochs > 90
        milestones = [50, 60, 70, 80,  90]
        scheduler = MultiStepLR(optimizer,  milestones=milestones, gamma=0.01)
    elif args.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=3e-7)
    else:
        raise NotImplementedError("Not implemented scheduler")


    return model, optimizer, train_loader, test_loader, scheduler


def train_classifier(args, net, optimizer, dataloader, recorder):

    CEloss = nn.CrossEntropyLoss()
    net.train()
    for inputs, label, index in dataloader:
        inputs = inputs.cuda()
        label = label.cuda() 
        optimizer.zero_grad()
        outputs = net(inputs)

        record_dynamics(index,  outputs.cpu().data, label.cpu().data, recorder, args.num_classes)
        
        loss = CEloss(outputs, label)
        loss.backward()
        optimizer.step()

def eval_classifier(net, loader):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, label in loader:
            data, label = Variable(data).cuda(), Variable(label).cuda()
            out = net(data)
            _, predicted = torch.max(out.data, dim=1)
            total += label.shape[0]
            correct += (predicted == label).sum().item()
    return correct / total



def generate_dynamics(args, saving_dynamics=True):
    
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    net, optimizer, train_loader, test_loader, scheduler = build_classifier(args)

    cudnn.benchmark = True

    recorder = [[] for _ in range(args.num_samples)]# recorder for training history of all samples
    best_acc = 0
    
    for epoch in range(args.num_epochs):
        # print("|Dynamics| epoch:%d" % epoch)
        train_classifier(args, net, optimizer, train_loader, recorder)
        scheduler.step()

        acc = eval_classifier(net, test_loader)

        if acc > best_acc:
            best_acc = acc

        print(f"|Dynamics| Epoch[{epoch+1:0>3}/{args.num_epochs:0>3}], best acc:{best_acc:.2%}, acc:{acc:.2%}")
    
    record_all = save_dynamics(recorder, train_loader, args.dynamics_path, saving_dynamics)
    
    return record_all
      


def train_cluster(args, net, optimizer, loader, L):
    net.train()
    l_cluster = 0.0
    l_sim = 0.0
    L_cos = torch.nn.CosineEmbeddingLoss()
    num_sim_total = 0
    for data, noise_label, cor_label, idx in loader:
        data, noise_label, cor_label, idx = Variable(data).cuda(), Variable(noise_label).cuda(), Variable(cor_label).cuda(), idx.cuda()
        z = net(data)
        
        dist = net.cos_similarity(z)  # N x 2
        
        q = net.get_q(dist)
        p = net.target_distribution(q.data)

        loss_cluster =  F.kl_div(q.log(), p, reduction='batchmean') # N
        

        ## sim loss
        if args.cor_rate > 0 :
            q0 =q[:,0].detach().cpu().numpy()
            noise_pred_bool = q0 > 0.5
            noise_pred_idx = idx.cpu().numpy()[noise_pred_bool]
            clean_pred_idx = idx.cpu().numpy()[~noise_pred_bool]

            
            org_n_pred = torch.from_numpy(np.intersect1d(L.org, noise_pred_idx)).to(args.device)
            cor_n_pred = torch.from_numpy(np.intersect1d(L.cor, noise_pred_idx)).to(args.device)
            org_c_pred = torch.from_numpy(np.intersect1d(L.org, clean_pred_idx)).to(args.device)
            cor_c_pred = torch.from_numpy(np.intersect1d(L.cor, clean_pred_idx)).to(args.device)


            if org_n_pred.shape[0] * cor_n_pred.shape[0] * org_c_pred.shape[0] * cor_c_pred.shape[0] > 0:               
                
                num_sim_data = org_n_pred.shape[0] + cor_n_pred.shape[0] + org_c_pred.shape[0] + cor_c_pred.shape[0]

                z_org_n_pred = torch.index_select(z, 0,  get_idx_of_idx(idx, org_n_pred, args.device))
                z_org_c_pred = torch.index_select(z, 0, get_idx_of_idx(idx, org_c_pred, args.device))
                z_cor_n_pred = torch.index_select(z, 0,  get_idx_of_idx(idx, cor_n_pred, args.device))
                z_cor_c_pred = torch.index_select(z, 0, get_idx_of_idx(idx, cor_c_pred, args.device))


                l_noise = L_cos(torch.mean(z_org_n_pred, dim=0), torch.mean(z_cor_n_pred, dim=0), torch.tensor(1.).to(args.device))
                l_clean = L_cos(torch.mean(z_org_c_pred, dim=0), torch.mean(z_cor_c_pred, dim=0), torch.tensor(1.).to(args.device))
                loss_sim = (l_noise + l_clean)/2 

            
            elif org_n_pred.shape[0] * cor_n_pred.shape[0] > 0:
                num_sim_data = org_n_pred.shape[0] + cor_n_pred.shape[0]            
                
                z_org_n_pred = torch.index_select(z, 0,  get_idx_of_idx(idx, org_n_pred, args.device))
                z_cor_n_pred = torch.index_select(z, 0,  get_idx_of_idx(idx, cor_n_pred, args.device))

                l_noise = L_cos(torch.mean(z_org_n_pred, dim=0), torch.mean(z_cor_n_pred, dim=0), torch.tensor(1.).to(args.device))

                loss_sim = l_noise  

            
            else:
                loss_sim=0
                num_sim_data = 0

        else:
            loss_sim = 0
            num_sim_data = 0

        loss = loss_cluster + args.alpha * loss_sim 

        optimizer.zero_grad()  
        loss.backward() 
        optimizer.step()
        num_data = data.shape[0]
        l_cluster += loss_cluster.item() * num_data   

        if type(loss_sim) != int:
            l_sim += loss_sim.item() 

        num_sim_total += num_sim_data
    
    if num_sim_total > 0:
        avg_l_sim = l_sim/num_sim_total
    else:
        avg_l_sim = 0

    return l_cluster/len(loader.dataset), avg_l_sim


def eval_basedon_pair(args, net, loader, L):
    net.eval()
    cor_labels = []
    noise_labels = []
    q0 = []
    zs = []
    with torch.no_grad():
        for data, noise_label, cor_label, _  in loader:
            data = Variable(data).cuda()

            z = net(data) 
            dist_sq = torch.sum(torch.pow(z.unsqueeze(1) - net.centers, 2), 2)  # N x 2
            q = net.get_q(dist_sq)

            cor_labels.append(cor_label)
            noise_labels.append(noise_label)
            q0.append(q[:, 0])
            zs.append(z.cpu())
    cor_labels = torch.cat(cor_labels, dim=0).numpy()
    noise_labels = torch.cat(noise_labels, dim=0).numpy()
    q0 = torch.cat(q0, dim=0).cpu().numpy()
    zs = torch.cat(zs, dim=0).cpu().numpy()
    
    noise_pred_bool = q0 > 0.5
    pred = noise_pred_bool.astype(int)
    
    f1 = f1_score(noise_labels[L.org], pred[L.org])

    
    if args.cor_rate == 0:
        metric = 0
        return f1, metric
    
    else:
        cor_n_pred = np.intersect1d(L.cor, bool2idx(noise_pred_bool))
        org_c_pred = np.intersect1d(L.org_pair_cor, bool2idx(~noise_pred_bool))
        
        if cor_n_pred.shape[0] * org_c_pred.shape[0] > 0:
            n_mean = np.mean(q0[cor_n_pred])
            c_mean = np.mean(q0[org_c_pred])        
            metric = (n_mean - c_mean)**2
        else:
            metric = 0


    return f1, metric




def run_clustering(args, recorder):
    
    setup_seed(args.seed)

    dynamics, class_label_clean, class_label_noise, logit_idx = load_dynamics(recorder) 

    cor_label, noise_label = get_label(args, class_label_clean, class_label_noise)

    L = lable_inform(cor_label, noise_label, logit_idx)


    dataset = cluster_dataset(dynamics, noise_label.tolist(), cor_label.tolist())
    org_loader = get_loader(dataset, 'original', shuffle=False, num_workers=args.num_workers)
    cor_loader = get_loader(dataset, 'corrupt', shuffle=False, num_workers=args.num_workers)
    all_loader_for_eval = get_loader(dataset, 'all', shuffle=False, num_workers=args.num_workers)
    all_loader = get_loader(dataset, 'all', shuffle=True, drop_last=True, batch_size=1024, num_workers=args.num_workers)

    f1_inits = 0
    metric_init = 0
    optimal_f1 = 0
    optimal_epoch = 0
        
    net = oneD_CNN_cluster(dynamics.shape[-1], dynamics.shape[1], args.rep_dim)
    net = net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.00001, weight_decay=0.0001)
    
    c_noisy, c_clean  = init_centroid(args, net, cor_loader, org_loader)
    net.centers.data = torch.cat((c_noisy, c_clean), dim=0).to(net.centers.device) # 2 x emb_dim 
    f1_init, metric = eval_basedon_pair(args, net, all_loader_for_eval, L)    

    f1_inits += f1_init
    metric_init += metric
    epoch_best = 0
    metric_best = metric
    f1_best = f1_init

    
    for epoch in range(1, args.num_epochs_cluster + 1):

        l_cluster, l_sim = train_cluster(args, net, optimizer, all_loader, L)

        f1, metric = eval_basedon_pair(args, net, all_loader_for_eval, L)
        

        if f1> optimal_f1:
            optimal_f1 = f1
            optimal_epoch = epoch
        if metric > metric_best:
            metric_best = metric
            epoch_best = epoch
            f1_best = f1
        
        print(f'|Clustering| {epoch},f1:{f1:.3f}, metric:{metric:.5f}, loss_cluster:{l_cluster:.5f},loss_sim:{l_sim:.5f}, optimal_f1:{optimal_f1*100:.2f}@{optimal_epoch} # Best Epoch {epoch_best}, metric: {metric_best:.5f}, F1:{f1_best*100:2f}')
        


def main(args):

    
    ######## Generation of input representations for classifier      
    if args.net == 'clip' and not os.path.exists(args.clip_rep_path):
        generate_clip_rep(args, args.clip_rep_path)
    elif 'resnet' in args.net:
        dir_path = os.path.join(args.save_dir, args.dataset, args.net)
        confirm_dir(dir_path)
        args.modified_traindata_path = os.path.join(dir_path, f'img_crop{args.resent_crop_size}.pth')
        if not os.path.exists(args.modified_traindata_path):
            generate_resnet_rep_for_cor(args, args.modified_traindata_path, args.resent_crop_size)

    ######## Generation of training dynamics

    if os.path.exists(args.dynamics_path): 
        with open(args.dynamics_path, "rb") as f1_val:
            dynamics_record = pickle.load(f1_val) 
    else:
        dynamics_record = generate_dynamics(args, saving_dynamics=True)


    ######## clustering 
    run_clustering(args, dynamics_record)



if __name__ == "__main__":
    args = load_args()

    print(args)

    main(args)

  




    
