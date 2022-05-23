import numpy as np
import torch
from scipy.interpolate import interp1d

def noise_channel(ts, degree):
    """
    Add noise to ts
    
    mode: high, low, both
    degree: degree of noise, compared with range of ts    
    
    Input:
        ts: (n_length)
    Output:
        out_ts: (n_length)
        
    """
    len_ts = len(ts)
    num_range = np.ptp(ts)+1e-4 # add a small number for flat signal
    
    noise1 = degree * num_range * (2*np.random.rand(len_ts)-1)
    noise2 = degree * num_range * (2*np.random.rand(len_ts//100)-1)
    x_old = np.linspace(0, 1, num=len_ts//100, endpoint=True)
    x_new = np.linspace(0, 1, num=len_ts, endpoint=True)
    f = interp1d(x_old, noise2, kind='linear')
    noise2 = f(x_new)
    out_ts = ts + noise1 + noise2

    return out_ts

def jitter(x, degree_scale=1):

  
    ret = []
    for chan in range(x.shape[0]):
        ret.append(noise_channel(x[chan], 0.05*degree_scale))
    ret = np.vstack(ret)
    ret = torch.from_numpy(ret)
    return ret 

def scaling(x,degree_scale=2):
    #eprint(x.shape)
    ret = np.zeros_like(x)
    degree = 0.05*(degree_scale+np.random.rand())
    factor = 2*np.random.normal(size=x.shape[1])-1
    factor = 1.5+(2*np.random.rand())+degree*factor
    for i in range(x.shape[0]):
        ret[i]=x[i]*factor
    ret = torch.from_numpy(ret)
    return ret 

def masking(x):
    # for each modality we are using different masking
    segments = 50 + int(np.random.rand()*(200-50))
    points = np.random.randint(0,3000-segments)
    ret = x.detach().clone()
    for i,k in enumerate(x):
        ret[i,points:points+segments] = 0

    return ret

def flip(x):
    # horizontal flip
    if np.random.rand() >0.5:
        return torch.tensor(np.flip(x.numpy(),1).copy())
    else:
        return x


def augment(x):
    ''' use jitter in every aug to get two different eeg signals '''
    weak_ret = masking(jitter(x))
    strong_ret = scaling(flip(x),degree_scale=3)
    return weak_ret,strong_ret