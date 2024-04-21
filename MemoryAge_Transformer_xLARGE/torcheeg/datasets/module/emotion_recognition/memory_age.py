import os
import pickle as pkl
from typing import Any, Callable, Dict, Tuple, Union

from ..base_dataset import BaseDataset


class MemoryAgeDataset(BaseDataset):
    r'''
    Args:
        root_path (str): raw data files in pickled python/numpy formats

        # ----- for IO -----
        io_path (str): The path to generated unified data IO, cached as an intermediate result.
                       'Unified data IO' is the place to save the processed data, which is used to train the model. For example:
                       - subfolder 'cross_subject_k_fold_split': the detailed info of train/test split for each fold, in csv format
                       - subfolder 'samples': each sample is in one file, in pickle format. There could be 100,000+ samples in one dataset.
                       - csv file 'info.csv': the meta info of each sample, such as subject_id, clip_id, label, trial_id
                                              note that in the TRANSFORMER project, label and trial_id are the same.
                                              the only reason we keep trial_id is becasue 'k_fold_cross_subject.py' requires it.
        io_size (int): Maximum size database may grow to; used to size the memory mapping. If database grows larger than ``map_size``,
                       an exception will be raised and the user must close and reopen. (default: :obj:`10485760`)
        io_mode (str): Storage mode of EEG signal. When io_mode is set to :obj:`lmdb`, TorchEEG provides an efficient database (LMDB) for storing EEG signals.
                       LMDB may not perform well on limited operating systems, where a file system based EEG signal storage is also provided.
                       When io_mode is set to :obj:`pickle`, pickle-based persistence files are used.
        num_worker (int): Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: :obj:`0`)
        verbose (bool): Whether to display logs during processing, such as progress bars, etc. (default: :obj:`True`)
        in_memory (bool): Whether to load the entire dataset into memory. 
                          If :obj:`in_memory` is set to True, then the first time an EEG sample is read, the entire dataset is loaded into memory for subsequent retrieval.
                          Otherwise, the dataset is stored on disk to avoid the out-of-memory problem. (default: :obj:`False`)

        # ----- in __getitem__() method                                        
        online_transform (Callable, optional): The transformation of the EEG signals. 
                                               The input is a :obj:`np.ndarray`, 
    '''

    def __init__(self,
                 root_path: str = './raw_dataset/data_memory_age',
                 io_path: str = './processed_dataset',
                 online_transform: Union[None, Callable] = None,    # used in __getitem__()
                 io_size: int = 10485760,
                 io_mode: str = 'pickle',
                 num_worker: int = 0,
                 verbose: bool = True,
                 in_memory: bool = False):

        params = {
            'root_path': root_path,
            'io_path': io_path,
            'online_transform': online_transform,
            'io_size': io_size,
            'io_mode': io_mode,
            'num_worker': num_worker,
            'verbose': verbose,
            'in_memory': in_memory
        }

        # excute the __init__() method of BaseDataset with the arguments in params
        super().__init__(**params)

        # save all arguments to __dict__
        self.__dict__.update(params)

    @staticmethod
    def _load_data(file: Any = None,
                   root_path: str = './raw_dataset/data_memory_age',
                   **kwargs):

        # load the data of one mouse in recent or remote session, in pickle format.
        # e.g., file = 'ym212_Recent'
        with open(os.path.join(root_path, file), 'rb') as f:
            samples = pkl.load(f)        # segments(~10000), channel(3), timestep(1600*2.56=4096)

        # loop for each segment
        for segment_id in range(len(samples)):

            chunk = samples[segment_id,:,:]  # channel(3), timestep(1600*2.56=4096)

            # --- record the common meta info ---
            subject_id = file.split('_')[0]          # e.g., 'ym212'
            chunk_info = {'subject_id': subject_id}
            # if the content after '_' in file is recent, then
            if file.split('_')[1] == 'Recent':
                chunk_info['clip_id'] = f'{subject_id}_Recent_{segment_id}'   # e.g., 'ym212_Recent_0'
                chunk_info['label'] = 0
                chunk_info['trial_id'] = 0
            else:
                chunk_info['clip_id'] = f'{subject_id}_Remote_{segment_id}'   # e.g., 'ym212_Remote_0'
                chunk_info['label'] = 1
                chunk_info['trial_id'] = 1
            # for example, chunk_info = {'subject_id': 'ym212', 'clip_id': 'ym212_Recent_589', 'label': 0, 'trial_id': 0}
            #              chunk_info = {'subject_id': 'ym212', 'clip_id': 'ym212_Remote_995', 'label': 1, 'trial_id': 1}
            # -----------------------------------

            yield {'sample': chunk, 'key': chunk_info['clip_id'], 'info': chunk_info}
            # chunk is an array with shape (3, 4096),
            # clip_id is a string, it is the name of the chunk when it is stored in the database
            # chunk_info is a dict, it contains the meta info of the chunk

    @staticmethod
    def _set_files(root_path: str = './raw_dataset/data_memory_age', **kwargs):
        return os.listdir(root_path)

    def __getitem__(self, index: int) -> Tuple:

        # read_info: self.info.iloc[index].to_dict()
        # self.info is a dataframe, it contains the meta info of all chunks, defined in base_dataset.py
        info = self.read_info(index)

        eeg_index = str(info['clip_id'])
        eeg = self.read_eeg(eeg_index)
        label = info['label']

        if self.online_transform:
            signal = self.online_transform(eeg=eeg)['eeg']

        return signal, label

    @property
    def repr_body(self) -> Dict:
        return dict(
            super().repr_body, **{
                'root_path': self.root_path,
                'online_transform': self.online_transform,
                'num_worker': self.num_worker,
                'verbose': self.verbose,
                'io_size': self.io_size
            })
