import os
import re
from copy import copy
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn import model_selection
from torcheeg.datasets.module.base_dataset import BaseDataset


class KFoldCrossSubject:
    r'''
    A tool class for k-fold cross-validations, to divide the training set and the test set. 
    One of the most commonly used data partitioning methods, where the data set is divided into k subsets of subjects, 
    with one subset subjects being retained as the test set and the remaining k-1 subset subjects being used as training data. 
    In most of the literature, K is chosen as 5 or 10 according to the size of the data set.

    Args:
        n_splits (int): Number of folds. Must be at least 2. (default: :obj:`5`)
        shuffle (bool): Whether to shuffle the data before splitting into batches. 
                        Note that the samples within each split will not be shuffled. (default: :obj:`False`)
                        - EXPLANATION: In our KFoldCrossSubject class, we use model_selection.KFold to spilt subject_ids into k folds.
                                       If shuffle is :obj:`True`, the order of subject_ids will be shuffled before splitting into batches.
        random_state (int, optional): When shuffle is :obj:`True`, :obj:`random_state` affects the ordering of the indices, 
                                      which controls the randomness of each fold. Otherwise, this parameter has no effect. 
                                      (default: :obj:`None`)
        split_path (str): The path to data partition information. 
                          If the path exists, read the existing partition from the path. 
                          If the path does not exist, the current division method will be saved for next use. 
    '''
    
    def __init__(self,
                 n_splits: int = 5,
                 shuffle: bool = False,
                 random_state: Union[None, int] = None,
                 split_path: str = './processed_dataset/cross_subject_k_fold'):
        
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.split_path = split_path

        self.k_fold = model_selection.KFold(n_splits=n_splits,
                                            shuffle=shuffle,
                                            random_state=random_state)
        # Note: The KFold class in sklearn.model_selection is used here,
        #       This KFold is not designed for 'cross-subject' data partitioning.
        #       But how to use this KFold class to do 'cross-subject' data partitioning?
        #       You will see that self.k_fold.split(subject_ids) is used in the split_info_constructor() method.
        #       BASICALLY, self.k_fold spilts 'subject_ids' to ensure 'CROSS-SUBJECT'

    def split_info_constructor(self, info: pd.DataFrame) -> None:
        '''
        - info is a df, it contains the meta info of all the samples from all the subjects in the dataset
          the columns of info are: ['subject_id', 'clip_id'(the name of the sample), 'label', 'trial_id']
        - Each time in the for loop, we sperate the info into training and testing (according to the current subject_ids spilt), 
          and save them into csv files. for example, train_fold_0.csv and test_fold_0.csv
        - Basically, we get k pairs of train/test csv files, and save them into the split_path
        '''

        # all the distinct subject ids, e.g. ['ym212', 'ym213',...]
        subject_ids = sorted(list(set(info['subject_id'])))      
        # note, sorted is a MUST after set. otherwise, the order of subject_ids will be different in different runs
        # subject_ids = 'ym212', 'ym213', ... , 'ym227'

        # split the subject ids into k folds using the KFold class in sklearn.model_selection
        # what we get are the indices of the subject_ids list
        for fold_id, (train_index_subject_ids, test_index_subject_ids) in enumerate(self.k_fold.split(subject_ids)):

            if len(train_index_subject_ids) == 0 or len(test_index_subject_ids) == 0:
                raise ValueError(f'The number of training or testing subjects is zero.')

            # get the subject ids of training and testing
            train_subject_ids = np.array(subject_ids)[train_index_subject_ids].tolist()
            test_subject_ids = np.array(subject_ids)[test_index_subject_ids].tolist()

            # get the info (df) of training and testing
            train_info = []
            for train_subject_id in train_subject_ids:
                train_info.append(info[info['subject_id'] == train_subject_id])
            train_info = pd.concat(train_info, ignore_index=True)

            test_info = []
            for test_subject_id in test_subject_ids:
                test_info.append(info[info['subject_id'] == test_subject_id])
            test_info = pd.concat(test_info, ignore_index=True)

            # save the info (df) of training and testing
            train_info.to_csv(os.path.join(self.split_path, f'train_fold_{fold_id}.csv'), index=False)
            test_info.to_csv(os.path.join(self.split_path, f'test_fold_{fold_id}.csv'), index=False)


    @property
    def fold_ids(self):
        '''
        this property returns a sorted list of unique fold IDs present in the directory specified by self.split_path. 
        These fold IDs are extracted from the names of .csv files that match the pattern fold_<some_number>.csv.
        '''
        indice_files = list(os.listdir(self.split_path))

        def indice_file_to_fold_id(indice_file):
            return int(re.findall(r'fold_(\d*).csv', indice_file)[0])

        fold_ids = list(set(map(indice_file_to_fold_id, indice_files)))
        fold_ids.sort()
        return fold_ids

    def split(self, dataset: BaseDataset) -> Tuple[BaseDataset, BaseDataset]:
        '''
        dataset is the object of a MemoryAgeDataset class
        '''
        if not os.path.exists(self.split_path):
            os.makedirs(self.split_path)
            
        # change in 20230803
        self.split_info_constructor(dataset.info)
        # dataset.info is a df, it contains the meta info of all the samples from all the subjects in the dataset
        # it contains columns: ['subject_id', 'clip_id'(the name of the sample), 'label', 'trial_id']

        fold_ids = self.fold_ids

        for fold_id in fold_ids:
            # read the meta info of training and testing from the csv files
            train_info = pd.read_csv(os.path.join(self.split_path, f'train_fold_{fold_id}.csv'))
            test_info = pd.read_csv(os.path.join(self.split_path, f'test_fold_{fold_id}.csv'))

            # build MemoryAgeDataset objects for training and testing
            # Note that the datasets are full copy of the original dataset. However, the info is specified to the current fold.
            # - When using PyTorch's DataLoader, the underlying dataset object should define the __getitem__ and __len__ methods to 
            #   fetch individual samples and to specify the dataset's length, respectively. If your train_dataset and test_dataset 
            #   are copies of the original dataset but with different info attributes, the impact on data loading depends on 
            #   how these __getitem__ and __len__ methods are implemented in your BaseDataset class.
            # - If the __getitem__ and __len__ methods of your BaseDataset class rely on the info attribute to determine what data 
            #   to return and how many samples there are, then changing info will essentially create two different datasets from the 
            #   same original dataset. The DataLoader will then generate batches according to these separate info settings. 
            # - __len__ in the BaseDataset class is implemented as len(self.info)
            # - __getitem__ in the child class MemoryAgeDataset also relies on the info attribute 
            train_dataset = copy(dataset)
            train_dataset.info = train_info

            test_dataset = copy(dataset)
            test_dataset.info = test_info

            # ----- print the info of the current fold -----
            train_subjects = list(set(train_info['subject_id']))
            test_subjects = list(set(test_info['subject_id']))
            print(f'-------------------- FOLD {fold_id} -------------------')
            print('train_subjects:', train_subjects)
            print('test_subjects:', test_subjects)
            print('------------------------------------------------------')
            
            # ----- yield the datasets of the current fold, stop and restart for the next fold -----
            yield train_dataset, test_dataset


    @property
    def repr_body(self) -> Dict:
        return {
            'n_splits': self.n_splits,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'split_path': self.split_path
        }

    def __repr__(self) -> str:
        # init info
        format_string = self.__class__.__name__ + '('
        for i, (k, v) in enumerate(self.repr_body.items()):
            # line end
            if i:
                format_string += ', '
            # str param
            if isinstance(v, str):
                format_string += f"{k}='{v}'"
            else:
                format_string += f"{k}={v}"
        format_string += ')'
        return format_string
