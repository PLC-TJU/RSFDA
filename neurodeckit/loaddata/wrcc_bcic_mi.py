"""
cross-session motor imagery dataset from Pan et al 2023
Authors: LC.Pan <panlincong@tju.edu.cn>
Date: 2024/7/5
License: BSD 3-Clause License
"""

import logging
import os

import mne
import numpy as np
from pooch import retrieve
from scipy.io import loadmat

from moabb.datasets.base import BaseDataset
from moabb.datasets.download import get_dataset_path

log = logging.getLogger(__name__)

# 2024A, 2023B
ID_List_2024A = [10358857, 10358855, 10358853, 10358856, 10358859,
                 10366513, 10358861, 10358854, 10358860] 

ID_List_2024B = [10466315, 10466310, 10466317, 10466311, 10466312,
                 10466316, 10466309, 10466314, 10466313] 

# 2023A, 2023B, 2023C
ID_List_2023A = [10358829, 10358821, 10358827, 10358828, 10358824,
                 10358822, 10358825, 10358823, 10358826] 

ID_List_2023B = [10358839, 10358833, 10358840, 10358836, 10358838,
                 10358834, 10358835, 10358837, 10358832] 

ID_List_2023C = [10358845, 10358849, 10358843, 10358842, 10358844,
                 10358847, 10358846, 10358848]

FILES_2024A = [f"https://dataverse.harvard.edu/api/access/datafile/{i}" for i in ID_List_2024A]
FILES_2024B = [f"https://dataverse.harvard.edu/api/access/datafile/{i}" for i in ID_List_2024B]
FILES_2023A = [f"https://dataverse.harvard.edu/api/access/datafile/{i}" for i in ID_List_2023A]
FILES_2023B = [f"https://dataverse.harvard.edu/api/access/datafile/{i}" for i in ID_List_2023B]
FILES_2023C = [f"https://dataverse.harvard.edu/api/access/datafile/{i}" for i in ID_List_2023C]

def eeg_data_path(subject, base_path='', phase='2024A'):
    """Load EEG data for a given subject from the BCIC dataset.

    Parameters
    ----------
    subject : int
        Subject number, mostly between 1 and 9. 
    phase : str, optional
        Phase of the dataset, either '2024A', '2023A', '2023B', or '2023C'. Defaults to '2024A'.
    base_path : str, optional
        Base path where the EEG data files are stored. Defaults to current directory.

    Returns
    -------
    list of str
        Paths to the subject's EEG data files.
    """
    if phase == '2024A':
        Files = FILES_2024A
    elif phase == '2024B':
        Files = FILES_2024B
    elif phase == '2023A': 
        Files = FILES_2023A
    elif phase == '2023B':
        Files = FILES_2023B
    elif phase == '2023C':
        Files = FILES_2023C
    else:
        raise ValueError(f"Invalid phase: {phase}.")
    
    if not 1 <= subject <= len(Files):
        raise ValueError(f"Subject must be between 1 and {len(Files)}. Got {subject}.")

    # 使用列表推导式和循环来简化文件下载逻辑
    file_path = os.path.join(base_path, f"Subject{str(subject).zfill(2)}.mat")

    # 下载缺失的文件
    if not os.path.isfile(file_path):
        url = Files[subject-1]
        retrieve(url, None, file_path, base_path, progressbar=True)

    return [file_path]

class BCIC_MI(BaseDataset):
    """Motor Imagery dataset from World Robot Contest Championships (WRCC) 2023/2024."""
    def _get_single_subject_data(self, subject):
        """Return data for a single subject."""
           
        montage = mne.channels.make_standard_montage("standard_1005")

        # fmt: off
        ch_names = [
            "Fpz", "Fp1", "Fp2", "AF3", "AF4", "AF7", "AF8", 
            "Fz", "F1", "F2", "F3", "F4", "F5", "F6","F7", "F8", 
            "FCz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6", "FT7", "FT8",
            "Cz", "C1", "C2", "C3", "C4", "C5", "C6", "T7", "T8", 
            "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "TP7", "TP8", 
            "Pz", "P3", "P4", "P5", "P6", "P7", "P8", 
            "POz", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8", 
            "Oz", "O1", "O2",
        ]
        # fmt: on

        ch_types = ["eeg"] * 59

        info = mne.create_info(
            ch_names=ch_names + ["STIM014"], ch_types=ch_types + ["stim"], sfreq=1000
        )
        
        sessions = self.data_path(subject, path=self.path)
        out = {}
        for sess_ind, fname in enumerate(sessions):
            data = loadmat(fname, squeeze_me=True, struct_as_record=False, verify_compressed_data_integrity=False)
            event_ids = data["label"].ravel()
            raw_data = data["data"]
            # de-mean each trial
            raw_data = raw_data - np.mean(raw_data, axis=2, keepdims=True)
            raw_events = np.zeros((raw_data.shape[0], 1, raw_data.shape[2]))
            raw_events[:, 0, 0] = event_ids
            data = np.concatenate([1e-3 * raw_data, raw_events], axis=1)
            # add buffer in between trials
            log.warning(
                "Trial data de-meaned and concatenated with a buffer to create " "cont data"
            )
            zeroshape = (data.shape[0], data.shape[1], 50)
            data = np.concatenate([np.zeros(zeroshape), data, np.zeros(zeroshape)], axis=2)
            
            trialnum = int(data.shape[0]/3)
            
            out[str(sess_ind)] = {}
            for run_ind in range(3):
                
                raw = mne.io.RawArray(
                    # 30/50 trials per run/block
                    data=np.concatenate(list(data[trialnum*run_ind:trialnum*(run_ind+1), :, :]), axis=1), info=info, verbose=False
                )
                raw.set_montage(montage)   
            
                out[str(sess_ind)][str(run_ind)] = raw
            
        return out


class BCIC2024A(BCIC_MI):
    """Motor Imagery dataset for stage A of the WRCC2024.
    
    Contains data from several stroke patients (ID unknown).

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        BCIC2024A        9       59           3                 30  4s            1000Hz                     1
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    """

    def __init__(self, **kwargs):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=1,
            events=dict(
                left_hand=1,
                right_hand=2,
                feet=3,
            ),
            code="BCIC2024A",
            interval=[0, 4],
            paradigm="imagery",
            doi="",
            **kwargs,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        path = get_dataset_path("BCIC", path)
        basepath = os.path.join(path,"MNE-bcic2024a-data")
        if not os.path.isdir(basepath):
            os.makedirs(basepath)

        return eeg_data_path(subject, basepath, '2024A')
    
class BCIC2024B(BCIC_MI):
    """Motor Imagery dataset for stage B of the WRCC2024.
    
    Contains data from several stroke patients (ID unknown).

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        BCIC2024A        9       59           3                 30  4s            1000Hz                     1
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    """

    def __init__(self, **kwargs):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=1,
            events=dict(
                left_hand=1,
                right_hand=2,
                feet=3,
            ),
            code="BCIC2024B",
            interval=[0, 4],
            paradigm="imagery",
            doi="",
            **kwargs,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        path = get_dataset_path("BCIC", path)
        basepath = os.path.join(path,"MNE-bcic2024b-data")
        if not os.path.isdir(basepath):
            os.makedirs(basepath)

        return eeg_data_path(subject, basepath, '2024B')

class BCIC2023A(BCIC_MI):    
    """Motor Imagery dataset for stage A of the WRCC2023.
    
    Contains data from seven healthy subjects and two stroke patients (ID unknown).

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        BCIC2024A        9       59           3                 30  4s            1000Hz                     1
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    """

    def __init__(self, **kwargs):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=1,
            events=dict(
                left_hand=1,
                right_hand=2,
                feet=3,
            ),
            code="BCIC2023A",
            interval=[0, 4],
            paradigm="imagery",
            doi="",
            **kwargs,
        )

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:
            raise (ValueError("Invalid subject number"))
        path = get_dataset_path("BCIC", path)
        basepath = os.path.join(path,"MNE-bcic2023a-data")
        if not os.path.isdir(basepath):
            os.makedirs(basepath)

        return eeg_data_path(subject, basepath, '2023A')

class BCIC2023B(BCIC_MI):    
    """Motor Imagery dataset for stage B of the WRCC2023.
    
    Contains data from seven healthy subjects and two stroke patients (ID unknown).

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        BCIC2024A        9       59           3                 30  4s            1000Hz                     1
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    """
    
    def __init__(self, **kwargs):
        super().__init__(
            subjects=list(range(1, 10)),
            sessions_per_subject=1,
            events=dict(    
                left_hand=1,
                right_hand=2,
                feet=3,
            ),
            code="BCIC2023B",            
            interval=[0, 4],
            paradigm="imagery",
            doi="",
            **kwargs,
        )        

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:            
            raise (ValueError("Invalid subject number"))
        path = get_dataset_path("BCIC", path)
        basepath = os.path.join(path,"MNE-bcic2023b-data")
        if not os.path.isdir(basepath):
            os.makedirs(basepath)

        return eeg_data_path(subject, basepath, '2023B')


class BCIC2023C(BCIC_MI):    
    """Motor Imagery dataset for stage C of the WRCC2023.
    
    Contains data from six healthy subjects and two stroke patients (ID unknown).

    .. admonition:: Dataset summary

        =========  =======  =======  ==========  =================  ============  ===============  ===========
        Name         #Subj    #Chan    #Classes    #Trials / class  Trials len    Sampling rate      #Sessions
        =========  =======  =======  ==========  =================  ============  ===============  ===========
        BCIC2024A        8       59           3                 30  4s            1000Hz                     1
        =========  =======  =======  ==========  =================  ============  ===============  ===========

    """
    
    def __init__(self, **kwargs):
        super().__init__(
            subjects=list(range(1, 9)),
            sessions_per_subject=1,
            events=dict(    
                left_hand=1,
                right_hand=2,
                feet=3,
            ),
            code="BCIC2023C",            
            interval=[0, 4],
            paradigm="imagery",
            doi="",
            **kwargs,
        )        

    def data_path(
        self, subject, path=None, force_update=False, update_path=None, verbose=None
    ):
        if subject not in self.subject_list:            
            raise (ValueError("Invalid subject number"))
        path = get_dataset_path("BCIC", path)
        basepath = os.path.join(path,"MNE-bcic2023c-data")
        if not os.path.isdir(basepath):
            os.makedirs(basepath)

        return eeg_data_path(subject, basepath, '2023C')

