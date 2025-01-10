
### SCRIPT TO COMPUTE KID AND FID METRICS ###

import torch
import random
import numpy as np
import json
import os
from tqdm import tqdm
from torch.utils.data import Dataset
from torch import randint
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance
from dataset import ImageDataset4Val


def set_seed(seed):
    """ Set the seed for reproducibility

    Parameters
    ----------
    seed : int
        Seed to use
    """
    # Set the seed for CPU
    torch.manual_seed(seed)
    
    # If you are using CUDA, set the seed for GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # Set the seed for numpy
    np.random.seed(seed)
    
    # Set the seed for Python random module
    random.seed(seed)
    
    # Ensure deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accumulate_KID(kid, dataloader, device, flag):
    """ Accumulate the Kernel Inception Distance (KID) of a set of images

    Parameters
    ----------
    kid : KernelInceptionDistance
        KID model
    dataloader : torch.utils.data.DataLoader
        DataLoader of the images
    device : torch.device
        Device to use
    flag : bool
        True for real images, False for synthetic images
    """

    for i, imgs in tqdm(enumerate(dataloader)):
        imgs = imgs.to(device)
        kid.update(imgs, real=flag)


def accumulate_FID(fid, dataloader, device, flag):
    """ Accumulate the Frechet Inception Distance (FID) of a set of images

    Parameters
    ----------
    fid : FrechetInceptionDistance
        FID model
    dataloader : torch.utils.data.DataLoader
        DataLoader of the images
    device : torch.device
        Device to use
    flag : bool
        True for real images, False for synthetic images
    """

    for imgs in dataloader:
        imgs = imgs.to(device)
        fid.update(imgs, real=flag)



def estimate_KID(real_dataloader, syn_dataloader, subset_size, device):
    """ Estimate the Kernel Inception Distance (KID) between real and synthetic images

    Parameters
    ----------
    real_dataloader : torch.utils.data.DataLoader
        DataLoader of the real images
    syn_dataloader : torch.utils.data.DataLoader
        DataLoader of the synthetic images
    subset_size : int
        Number of images to use for the KID estimation
    device : torch.device
        Device to use

    Returns
    -------
    kid_mean : float
        Mean KID
    kid_std : float
        Standard deviation of the KID
    """

    kid = KernelInceptionDistance(subset_size=subset_size, normalize=True).to(device)
    accumulate_KID(kid, real_dataloader, device, flag=True)
    accumulate_KID(kid, syn_dataloader, device, flag=False)
    kid_mean, kid_std = kid.compute()
    kid.reset()
    return kid_mean, kid_std


def estimate_FID(real_dataloader, syn_dataloader, device):
    """ Estimate the Frechet Inception Distance (FID) between real and synthetic images

    Parameters
    ----------
    real_dataloader : torch.utils.data.DataLoader
        DataLoader of the real images
    syn_dataloader : torch.utils.data.DataLoader
        DataLoader of the synthetic images
    device : torch.device
        Device to use

    Returns
    -------
    fid : float
        FID
    """

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    accumulate_FID(fid, real_dataloader, device, flag=True)
    accumulate_FID(fid, syn_dataloader, device, flag=False)
    fid_score = fid.compute()
    fid.reset()
    return fid_score


def evaluate_FID_inf(N, device, real_imgs_dir, syn_imgs_dir, batch_size):
    """ Evaluate the FID for different number of images and extract the linear fit

    Parameters
    ----------
    N : list
        List of number of images to evaluate
    device : torch.device
        Device to use
    real_imgs_dir : str
        Path to the real images
    syn_imgs_dir : str
        Path to the synthetic images
    batch_size : int
        Batch size

    Returns
    -------
    fid_scores : dict
        Dictionary with the FID scores for each number of images and the linear fit
    """

    fid_scores = {}

    for n in tqdm(N):
        real_dataset = ImageDataset4Val(img_dir=real_imgs_dir, num_imgs=n, seed=random.randint(0, 1000), flag='real', with_mask=True)
        syn_dataset = ImageDataset4Val(img_dir=syn_imgs_dir, num_imgs=n, seed=random.randint(0, 1000), flag='fake', with_mask=True)
        real_dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=True)
        syn_dataloader = torch.utils.data.DataLoader(syn_dataset, batch_size=batch_size, shuffle=True)
        fid_score = estimate_FID(real_dataloader, syn_dataloader, device)
        fid_scores[n] = fid_score.item()

    # Linear fit
    x = np.array(list(fid_scores.keys())[::-1])
    x = 1 / x
    y = np.array(list(fid_scores.values())[::-1])
    m, q = np.polyfit(x, y, 1)
    fid_scores['slope'] = m
    fid_scores['intercept'] = q

    return fid_scores
    

    
    


if __name__ == '__main__':

    # Load configuration file
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Dataset parameters
    real_imgs_dir = config['data']['real_imgs_dir']
    gen_imgs_dir = config['data']['gen_imgs_dir']
    batch_size = config['data']['batch_size']
    device = torch.device(config['kid']['device'])

    # KID parameters
    N_kid = config['kid']['N']
    subset_size = config['kid']['subset_size']
    save_folder = config['kid']['save_folder']
    seed = config['data']['seed']
    set_seed(seed=seed) # set seed for reproducibility

    
    ## KID estimation
    print('Estimating KID...')

    kid_results = {'KID_mean' : [],
                   'KID_std' : [],
                   'subset_size' : subset_size,
                   'N_kid' : N_kid,
                   'batch_size' : batch_size,
                   'seed' : seed}

    for num_imgs_for_kid in N_kid:
        real_dataset = ImageDataset4Val(img_dir=real_imgs_dir, num_imgs=num_imgs_for_kid, seed=random.randint(0, 1000), flag='real', with_mask=True)
        syn_dataset = ImageDataset4Val(img_dir=gen_imgs_dir, num_imgs=num_imgs_for_kid, seed=random.randint(0, 1000), flag='fake', with_mask=True)
        real_dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=batch_size, shuffle=True)
        syn_dataloader = torch.utils.data.DataLoader(syn_dataset, batch_size=batch_size, shuffle=True)
        kid_mean, kid_std = estimate_KID(real_dataloader, syn_dataloader, subset_size=subset_size, device=device)
        kid_results['KID_mean'].append(kid_mean.item())
        kid_results['KID_std'].append(kid_std.item())


    # Save KID results
    filename = os.path.join(save_folder, 'kid_results.json') 
    with open(filename, 'w') as f:
        json.dump(kid_results, f, indent=4)



    ## FID infinity parameters
    N_fid = config['fid']['N']
    num_experiments = config['fid']['num_experiments']
    save_folder = config['fid']['save_folder']

    ## FID estimation
    print('Estimating FID infinity...')

    fid_results = {'N_fid' : N_fid,
                   'num_experiments' : num_experiments,
                   'batch_size' : batch_size,
                   'seed' : seed}

    for n in range(num_experiments):
        fid_scores = evaluate_FID_inf(N_fid, device, real_imgs_dir, gen_imgs_dir, batch_size)
        fid_results[n] = fid_scores

    # Save FID results
    filename = os.path.join(save_folder, 'fid_results.json')
    with open(filename, 'w') as f:
        json.dump(fid_results, f, indent=4)
    







