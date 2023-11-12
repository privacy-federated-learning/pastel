import os
import pandas as pd
import operator
import math
from copy import deepcopy
import numpy as np
import scipy

import random

from torch.utils.data import Dataset

from data.motionsense import readwrite


def load_voxceleb_item(fileid, path, ext_audio):
    min_f0 = 60
    max_f0 = 400
    frame_length = 35
    frame_shift = 10
    file_audio = os.path.join(path, fileid + ext_audio)
    speaker_id = fileid.split("/")[-1].split('-')[0]
    data = readwrite.read_raw_mat(file_audio, 1)
    return data, speaker_id, fileid.split("/")


class GeneralPurposeDataset(Dataset):
    _ext_audio = ".f0"

    def __init__(self, root, folder="", partition="", load_item_fn=load_voxceleb_item, ext_audio=".f0", **kwargs):

        self._ext_audio = ext_audio
        # self._load_kwargs = load_kwargs
        folder_in_archive = os.path.join(folder, partition)
        self._path = os.path.join(root, folder_in_archive)
        # print(f"path: {self._path}")
        walker = walk_files_general(self._path, suffix=self._ext_audio, prefix=True, remove_suffix=True)
        self._walker = list(walker)

        self._walker.sort()
        self._walker = [f for f in self._walker if f.split('/')[-1][0] != '.']
        self.load_item_fn = load_item_fn

        self.desc_df = None

    def __getitem__(self, n):
        fileid = self._walker[n]

        return self.load_item_fn(fileid, self._path, self._ext_audio)

    def __len__(self):
        return len(self._walker)

    def trim_dataset(self, start, end):

        self._walker = self._walker[start:end]

    def trim_dataset_index(self, inds):
        self._walker = [self._walker[i] for i in inds]

    def get_desc_df(self):
        indexed_files = [(i, f.split("/")[-1].split('_')[0], f.split("/")[-1].split('_')[1]) for i, f in
                         enumerate(self._walker)]
        df = pd.DataFrame.from_records(indexed_files, columns=['ind', 'user', 'class'])
        self.desc_df = df
        return df

    def get_clients(self):
        df = self.get_desc_df()
        if 'client' in df.columns:
            return list(df['client'].unique())
        df['client'] = [u.split('-')[0] for u in df['user'].values]
        return list(df['client'].unique())

    def filter_client(self, client):
        if client not in self.get_clients():
            return []
        df = self.desc_df
        df = df[df['client'] == client]
        inds = df['ind'].values
        self._walker = operator.itemgetter(*inds)(self._walker)
        return inds

    def split_clients(self):
        clients = self.get_clients()
        datasets = {}
        for c in clients:
            cp_data = deepcopy(self)
            cp_data.filter_client(c)
            datasets[c] = cp_data
        return datasets

    def split_fake_clients(self, nb):
        original_len = len(self)
        inds = list(range(original_len))
        random.shuffle(inds)
        if nb > original_len:
            client_len = 1
            nb = original_len
        else:
            client_len = int(original_len / nb)
        datasets = {}
        for c in range(1, 1 + nb):
            cp_data = deepcopy(self)
            sub_inds = inds[(c - 1) * client_len:c * client_len]
            cp_data.trim_dataset_index(sub_inds)
            datasets[str(c)] = cp_data
        return datasets

    def augment_class(self, class_name, min_prop):
        df = self.get_desc_df()
        sub_df = df[df['class'] == class_name]
        total = len(df)
        sub_total = len(sub_df)
        if sub_total / float(total) < min_prop:
            sub_inds = sub_df['ind'].values
            frac, whole = math.modf(min_prop / (sub_total / float(total)))
            multi_inds = df['ind']  # df[df['class']!=class_name]['ind'].values
            if 0 < int(whole - 1):
                multi_inds = np.concatenate((multi_inds, np.repeat(sub_inds, int(whole - 1))))
            if 0 < int(math.ceil(frac * sub_total)) < sub_total:
                multi_inds = np.concatenate((multi_inds, sub_inds[:int(math.ceil(frac * sub_total))]))
            self._walker = operator.itemgetter(*multi_inds)(self._walker)

    def subset_spk_id_train(self, prop):
        indexed_files = [(i, f.split("/")[-1].split('-')[0]) for i, f in enumerate(self._walker)]
        df = pd.DataFrame.from_records(indexed_files, columns=['ind', 'spk'])
        tt = df.groupby('spk').apply(lambda x: x.head(int(len(x) * prop))).reset_index(drop=True)
        inds = tt['ind'].values
        self._walker = operator.itemgetter(*inds)(self._walker)

    def subset_spk_id_val(self, prop):
        indexed_files = [(i, f.split("/")[-1].split('-')[0]) for i, f in enumerate(self._walker)]
        df = pd.DataFrame.from_records(indexed_files, columns=['ind', 'spk'])
        tt = df.groupby('spk').apply(lambda x: x.tail(int(len(x) * prop))).reset_index(drop=True)
        inds = tt['ind'].values
        self._walker = operator.itemgetter(*inds)(self._walker)

    def subset_classes_train(self, prop):
        indexed_files = [(i, f.split("/")[-1].split('_')[0], f.split("/")[-1].split('_')[1]) for i, f in
                         enumerate(self._walker)]
        df = pd.DataFrame.from_records(indexed_files, columns=['ind', 'user', 'class'])

        tt = df.groupby('class').apply(lambda x: x.head(int(len(x) * prop))).reset_index(drop=True)

        inds = tt['ind'].values
        self._walker = operator.itemgetter(*inds)(self._walker)

    def subset_classes_val(self, prop):
        indexed_files = [(i, f.split("/")[-1].split('_')[0], f.split("/")[-1].split('_')[1]) for i, f in
                         enumerate(self._walker)]
        df = pd.DataFrame.from_records(indexed_files, columns=['ind', 'user', 'class'])
        tt = df.groupby('class').apply(lambda x: x.tail(int(len(x) * prop))).reset_index(drop=True)
        inds = tt['ind'].values
        self._walker = operator.itemgetter(*inds)(self._walker)

    def subSet_spk(self, samp):
        indexed_files = [(i, f.split("/")[-1].split('-')[0]) for i, f in enumerate(self._walker)]
        df = pd.DataFrame.from_records(indexed_files, columns=['ind', 'spk'])
        spks = np.sort(df['spk'].unique())
        if samp < len(spks):
            sub_spks = np.random.choice(spks, samp, replace=False)
            df = df[df['spk'].isin(sub_spks)]
        inds = df['ind'].values
        self._walker = operator.itemgetter(*inds)(self._walker)
        return inds

    def sample_index(self, inds):
        self._walker = operator.itemgetter(*inds)(self._walker)

    def sample(self, prop):
        L = len(self._walker)
        inds = np.random.choice(L, int(prop * L), replace=False)
        self.sample_index(inds)

    def get_spks(self):
        indexed_files = [(i, f.split("/")[-1].split('-')[0]) for i, f in enumerate(self._walker[1:])]
        df = pd.DataFrame.from_records(indexed_files, columns=['ind', 'spk'])
        spks = df['spk'].unique()
        return spks

    def get_classes(self):
        data = enumerate(self._walker[1:])
        indexed_files = [(i, f.split("/")[-1].split('_')[0], f.split("/")[-1].split('_')[1]) for (i, f) in
                         enumerate(self._walker[1:])]
        df = pd.DataFrame.from_records(indexed_files, columns=['ind', 'user', 'class'])
        classes = df['class'].unique()
        return classes

    def get_users(self):
        indexed_files = [(i, f.split("/")[-1].split('_')[0], f.split("/")[-1].split('_')[1]) for i, f in
                         enumerate(self._walker)]
        df = pd.DataFrame.from_records(indexed_files, columns=['ind', 'user', 'class'])
        users = df['user'].unique()
        return users

    def subset_classes(self, sub_classes):
        indexed_files = [(i, f.split("/")[-1].split('_')[0], f.split("/")[-1].split('_')[1]) for i, f in
                         enumerate(self._walker)]
        df = pd.DataFrame.from_records(indexed_files, columns=['ind', 'user', 'class'])
        df = df[df['class'].isin(sub_classes)]
        inds = df['ind'].values
        self._walker = operator.iwtemgetter(*inds)(self._walker)
        return inds

    def subset_users(self, samp):
        indexed_files = [(i, f.split("/")[-1].split('-')[0], f.split("/")[-1].split('-')[1]) for i, f in
                         enumerate(self._walker)]
        df = pd.DataFrame.from_records(indexed_files, columns=['ind', 'user', 'class'])
        users = np.sort(df['user'].unique())
        if samp < len(users):
            sub_users = np.random.choice(users, samp, replace=False)
            df = df[df['user'].isin(sub_users)]
        inds = df['ind'].values
        self._walker = operator.itemgetter(*inds)(self._walker)
        return inds


def load_AF_stats_item(fileid, path, ext):
    file = os.path.join(path, fileid + ext)
    user_id = fileid.split("/")[-1].split('_')[0]

    class_id = fileid.split("/")[-1].split('_')[-1]

    data = scipy.io.loadmat(file)['val'][0]

    X_stats = np.array((np.mean(data), np.std(data), np.min(data), np.max(data),))

    meanf0 = np.mean(data)
    stdf0 = np.std(data)
    minf0 = np.min(data)
    maxf0 = np.max(data)

    grad1 = np.gradient(data)

    meangrad1 = np.mean(grad1)
    stdgrad1 = np.std(grad1)
    mingrad1 = np.min(grad1)
    maxgrad1 = np.max(grad1)

    grad2 = np.gradient(grad1)

    meangrad2 = np.mean(grad2)
    stdgrad2 = np.std(grad2)
    mingrad2 = np.min(grad2)
    maxgrad2 = np.max(grad2)

    X_stats = np.array(
        (meanf0,
         stdf0,
         minf0,
         maxf0,
         meangrad1,
         stdgrad1,
         mingrad1,
         maxgrad1,
         meangrad2,
         stdgrad2,
         mingrad2,
         maxgrad2)
    )

    return X_stats, class_id, fileid.split("/")


def walk_files_general(root, prefix=True, suffix=None, remove_suffix=False):
    root = os.path.expanduser(root)

    for folder, _, files in os.walk(root):
        for f in files:
            if suffix:
                if f.endswith(suffix):

                    if remove_suffix:
                        f = f[:-len(suffix)]
                    if prefix:
                        f = os.path.join(folder, f)

                    yield f
            else:
                if prefix:
                    f = os.path.join(folder, f)

                yield f
