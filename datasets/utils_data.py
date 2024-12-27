import errno
import hashlib
import os
import os.path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.testing import assert_array_almost_equal
import random


def check_integrity(fpath, md5):
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def download_url(url, root, filename, md5):
    from six.moves import urllib

    root = os.path.expanduser(root)
    fpath = os.path.join(root, filename)

    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(url, fpath)
        except:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(url, fpath)


# basic function#
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    # print np.max(y), P.shape[0]
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    # print m
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n
        # print(P)
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print P

    return y_train, actual_noise

def noisify_randomflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        randomly flip 
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        for i in range(nb_classes):
            a= list(range(nb_classes))  
            a.pop(i)
            pick = random.choice(a)
            P[i, i], P[i, pick] = 1. - n, n
        # print(P)
        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print P

    return y_train, actual_noise


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes - 1):
            P[i, i] = 1. - n
        P[nb_classes - 1, nb_classes - 1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        # print('Actual noise %.2f' % actual_noise)
        y_train = y_train_noisy
    # print P

    return y_train, actual_noise



def noisify_instance(train_data, train_labels, noise_rate): ## 50000 x 512
    if max(train_labels)>10:
        num_class = 100
    else:
        num_class = 10
    np.random.seed(0)

    q_ = np.random.normal(loc=noise_rate,scale=0.1, size=1000000)
    q = []
    for pro in q_:
        if 0 < pro < 1:
            q.append(pro)
        if len(q)==50000:
            break
    
    w_size = train_data[0].flatten().shape[0]

    w = []
    for i in range(num_class):
        w.append(np.random.normal(loc=0,scale=1, size=(w_size, num_class))) # C * (512 *C)

    noisy_labels = []
    T = np.zeros((num_class,num_class))
    for i, sample in enumerate(train_data): 
        sample = sample.flatten() # sample 512
        p_all = np.matmul(sample, w[train_labels[i]])  # (1 x512) x (512 x c) = C
        p_all[train_labels[i]] = -1000000
        p_all = q[i]* F.softmax(torch.tensor(p_all),dim=0).numpy()
        p_all[train_labels[i]] = 1 - q[i]
        noisy_labels.append(np.random.choice(np.arange(num_class),p=p_all/sum(p_all)))
        T[train_labels[i]][noisy_labels[i]] += 1
    over_all_noise_rate = 1 - float(torch.tensor(train_labels).eq(torch.tensor(noisy_labels)).sum())/50000
    T = T/np.sum(T,axis=1)
    print(np.round(T*100,1))
    return noisy_labels, over_all_noise_rate


def noisify(nb_classes, train_labels:list, noise_type=None, noise_rate=0, random_state=0):

    train_labels = np.expand_dims(train_labels, axis=1)

    if noise_type == 'asym':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    elif noise_type == 'sym':
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    elif noise_type == 'randomflip':
        train_noisy_labels, actual_noise_rate = noisify_randomflip(train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)    
    else:
        print(f'Current input is {noise_type}')
    
    train_noisy_labels = np.squeeze(train_noisy_labels, axis=1)
    return train_noisy_labels, actual_noise_rate




def get_random_idx(data, ratio, random_state=0):
    ''''
    data: number of data or index array
    corrupting 할 때 사용한 데이터와 겹치면 안되므로, test 데이터는 random state를 이용해 고정함
    '''
    if type(data) == int:
        num_data = data
        idx = np.arange(num_data)
    else:
        num_data = data.shape[0]
        idx = data
        
    num_select = int(num_data * ratio)
    Rand = np.random.RandomState(random_state)
    Rand.shuffle(idx)
    idx_selected = idx[:num_select]

    return idx_selected