
# coding: utf-8

# In[8]:


from __future__ import print_function


import argparse
import os
import shutil
import time
import random
import torch.nn.functional as F


import torch
import pickle
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from privacy_defense_mechanisms.relaxloss.source.utils import Logger, AverageMeter, accuracy,  savefig
from progress.bar import Bar
import numpy as np
import tarfile
from sklearn.cluster import KMeans
from sklearn import datasets
import urllib


DATASET_PATH='data'
DATASET_NAME= 'dataset_purchase'
DATASET_FILE = os.path.join(DATASET_PATH,DATASET_NAME)

if not os.path.isfile(DATASET_FILE):
	print("Dowloading the dataset...")
	urllib.request.urlretrieve("https://www.comp.nus.edu.sg/~reza/files/dataset_purchase.tgz",os.path.join(DATASET_PATH,'tmp.tgz'))
	print('Dataset Dowloaded')

	tar = tarfile.open(os.path.join(DATASET_PATH,'tmp.tgz'))
	tar.extractall(path=DATASET_PATH)


print("Generating_dataset")
data_set =np.genfromtxt(DATASET_FILE,delimiter=',')
X = data_set[:,1:].astype(np.float64)
Y = (data_set[:,0]).astype(np.int32)-1

print("Dataset Generated")

print("Processing dataset")
len_train =len(X)
print(len_train)
# input(X)
r=np.load(open('/home2/cboscher/code/pastel2.0/data/purchase/random_r_purchase100.npy', 'rb'))
print(r)
X=X[r]
Y=Y[r]

print("R loaded")
train_classifier_ratio, train_attack_ratio = 0.1,0.15
train_classifier_data = X[:int(train_classifier_ratio*len_train)]
train_attack_data = X[int(train_classifier_ratio*len_train):int((train_classifier_ratio+train_attack_ratio)*len_train)]
test_data = X[int((train_classifier_ratio+train_attack_ratio)*len_train):]

train_classifier_label = Y[:int(train_classifier_ratio*len_train)]
train_attack_label = Y[int(train_classifier_ratio*len_train):int((train_classifier_ratio+train_attack_ratio)*len_train)]
test_label = Y[int((train_classifier_ratio+train_attack_ratio)*len_train):]


np.save('/home2/cboscher/code/pastel2.0/data/purchase/purchase_train_data.npy',train_classifier_data)
np.save('/home2/cboscher/code/pastel2.0/data/purchase/purchase_test_data.npy',test_data)
np.save('/home2/cboscher/code/pastel2.0/data/purchase/purchase_train_labels.npy',train_classifier_label)
np.save('/home2/cboscher/code/pastel2.0/data/purchase/purchase_test_labels.npy',test_label)
print("OK")