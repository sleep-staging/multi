from augmentations import *
from loss import loss_fn
from model import sleep_model
from train import *
from utils import *

from braindecode.util import set_random_seeds

import os
import numpy as np
import copy
import wandb
import torch
from torch.utils.data import DataLoader, Dataset

def main():
    
    PATH = '/scratch/shhs_3'
    SLEEPEDF_PATH = '/scratch/sleepedf_7'

    # Params
    SAVE_PATH = "me-shhs-3.pth"
    WEIGHT_DECAY = 1e-4
    BATCH_SIZE = 128 
    lr = 5e-4
    n_epochs = 400
    NUM_WORKERS = 6
    N_DIM = 256
    TEMPERATURE = 1

    ####################################################################################################

    random_state = 1234

    # Seeds
    rng = np.random.RandomState(random_state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False
        print(f"GPU available: {torch.cuda.device_count()}")

    set_random_seeds(seed=random_state, cuda=device == "cuda")


    ##################################################################################################


    # Extract number of channels and time steps from dataset
    n_channels, input_size_samples = (1, 3000)
    model = sleep_model(n_channels, input_size_samples, n_dim = N_DIM)

    q_encoder = model.to(device)

    optimizer = torch.optim.Adam(q_encoder.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = loss_fn(device, T=TEMPERATURE).to(device)

    #####################################################################################################

    class pretext_data(Dataset):

        def __init__(self, filepath):       
            self.file_path = filepath
            self.idx = np.array(range(len(self.file_path)))

        def __len__(self):
            return len(self.file_path)

        def __getitem__(self, index):
            
            path = self.file_path[index]
            data = np.load(path)
            pos = data['pos'][:, :1, :] # (7, 1, 3000)
            anc = copy.deepcopy(pos)
            
            # augment
            for i in range(pos.shape[0]):
                pos[i] = augment(pos[i])
                anc[i] = augment(anc[i])
            return anc, pos
      

    PRETEXT_FILE = os.listdir(os.path.join(PATH, "pretext"))
    PRETEXT_FILE.sort(key=natural_keys)
    PRETEXT_FILE = [os.path.join(PATH, "pretext", f) for f in PRETEXT_FILE]

    TEST_FILE = os.listdir(os.path.join(SLEEPEDF_PATH, "test"))
    TEST_FILE.sort(key=natural_keys)
    TEST_FILE = [os.path.join(SLEEPEDF_PATH, "test", f) for f in TEST_FILE]

    print(f'Number of pretext files: {len(PRETEXT_FILE)}')
    print(f'Number of test records: {len(TEST_FILE)}')

    pretext_loader = DataLoader(pretext_data(PRETEXT_FILE), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

    test_records = [np.load(f) for f in TEST_FILE]
    test_subjects = dict()

    for i, rec in enumerate(test_records):
        if rec['_description'][0] not in test_subjects.keys():
            test_subjects[rec['_description'][0]] = [rec]
        else:
            test_subjects[rec['_description'][0]].append(rec)

    test_subjects = list(test_subjects.values())


    ##############################################################################################################################


    wb = wandb.init(
            project="EPF-V2",
            notes="multi-epoch, symmetric loss, using same projection heads and no batch norm",
            save_code=True,
            entity="sleep-staging",
            name="me-shhs-3, transfer, new aug",
        )
    wb.save('multi/me_transfer/*.py')
    wb.watch([q_encoder],log='all',log_freq=500)

    Pretext(q_encoder, optimizer, n_epochs, criterion, pretext_loader, test_subjects, wb, device, SAVE_PATH, BATCH_SIZE)

    wb.finish()


if __name__ == "__main__":
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    main()