import os
import sys
sys.path.append(os.getcwd())

import cv2
import numpy as np
import pickle

import torch
import torchvision
from recognition.dataset.vgg_face2 import vggDataset
from recognition.estimator.evm import EVM

def pickle_load(path):
    with open(path, 'rb') as pickle_in:
        return pickle.load(pickle_in)

if __name__ == '__main__':
    dataset = vggDataset('/media/Datasets/vgg_face2') ; train = dataset.train_images
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    train = pickle_load('vgg_train.pkl')

    
    

    reload_evm=False
    evm = EVM()
    if reload_evm:
        with open('evm.pkl', 'rb') as pickle_in:
            evm = pickle.load(pickle_in)

    '''
    for i, data in enumerate(dataloader, 0):
        X,Y = data
        print(X, Y)

        evm.fit(X.numpy(),Y.numpy())
        print('[%d] fit' % (i))
    '''

    X = np.array(list(map(lambda x: x['descriptor'], train)))
    Y = np.array(list(map(lambda x: np.array(x['label']), train)))
    print(X.shape, Y.shape)
    evm.fit(X, Y)
    
    with open('evm.pkl', 'wb') as pickle_out:
        pickle.dump(evm, pickle_out, pickle.HIGHEST_PROTOCOL)
