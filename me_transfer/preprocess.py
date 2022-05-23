import os
import numpy as np
from tqdm import tqdm

EPOCH_LEN = 7

random_state = 1234
np.random.seed(random_state)
rng = np.random.RandomState(random_state)

PATH = '/scratch/shhs_7/'
DATA_PATH = os.path.join(PATH, "shhs_outputs")    
files = os.listdir(DATA_PATH)
files = np.array([os.path.join(DATA_PATH, i) for i in files])
files.sort()


######## pretext files##########
pretext_files = list(rng.choice(files, 264, replace=False))   
print("pretext files: ", len(pretext_files))

# load files
half_window = EPOCH_LEN // 2
os.makedirs(PATH+"/pretext/", exist_ok=True)

cnt = 0
for file in tqdm(pretext_files):
    x_dat = np.load(file)["x"]
    
    if x_dat.shape[-1]==2:
        x_dat = x_dat.transpose(0,2,1)

        for i in range(half_window, x_dat.shape[0] - half_window):
            dict = {}
            temp_path = os.path.join(PATH+"/pretext/", str(cnt)+".npz")
            dict['pos'] = x_dat[i-half_window : i+half_window+1]
            np.savez(temp_path, **dict)
            cnt+=1


######## test files##########
test_files = sorted(list(set(files)-set(pretext_files))) 
os.makedirs(PATH+"/test/",exist_ok=True)

print("test files: ", len(test_files))

for file in tqdm(test_files):
    new_dat = {}
    dat = np.load(file)

    if dat['x'].shape[-1]==2:
        new_dat['x'] = dat['x'].transpose(0,2,1)
        new_dat['y'] = dat['y']
        
        temp_path = os.path.join(PATH+"/test/",os.path.basename(file))
        np.savez(temp_path,**new_dat)