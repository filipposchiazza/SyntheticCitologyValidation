import os
import torch
import random
import torch.utils.data as data
from torchvision.io import read_image



class ImageDataset4Val(data.Dataset):

    def __init__(self, img_dir, num_imgs, seed, flag='real', with_mask=True):
        """Image dataset for validation.
        
        Parameters
        ----------
        img_dir : str
            Path to the folder containing the images.
        flag : str
            'real' or 'fake'.
        with_mask : bool
            Whether the masks are included in the images.
        """
        self.img_dir = img_dir
        self.num_imgs = num_imgs
        self.seed = seed
        self.flag = flag
        self.with_mask = with_mask
        img_filenames = sorted(os.listdir(img_dir))
        random.seed(seed)
        self.img_filenames = random.sample(img_filenames, num_imgs)

    def __len__(self):
        return len(self.img_filenames)
        
    def __getitem__(self, idx):
        img_name = self.img_filenames[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = read_image(img_path)[:3, :, :]
        if self.with_mask == True:
            img = img[:, :, :img.shape[2] // 2]
        img = img / 255.0
        return img

