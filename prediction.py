import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image

from torch_em.transform.raw import standardize as normlize
import models_mae_infer
import imageio.v2 as imageio
from torchvision import transforms
from torchvision.utils import save_image

livecell_mean = 128.0231
livecell_std = 10.6299

def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.mean(1.0*image,axis=-1,keepdims=True)) 
    #plt.imshow(torch.clip((image * livecell_std + livecell_mean) * 1, 0, 1).int())    #
    plt.title(title, fontsize=16)
    plt.axis('off')
    return

def prepare_model(chkpt_dir, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae_infer, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model

def run_one_image(img, model):
    x = torch.tensor(img)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    loss, y, mask = model(x.float(), mask_ratio=0.50)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    # make the plt figure larger
    plt.rcParams['figure.figsize'] = [24, 24]

    plt.subplot(1, 4, 1)
    show_image(x[0], "original")

    plt.subplot(1, 4, 2)
    show_image(im_masked[0], "masked")

    plt.subplot(1, 4, 3)
    show_image(y[0], "reconstruction")

    plt.subplot(1, 4, 4)
    show_image(im_paste[0], "reconstruction + visible")

    plt.show()
    plt.savefig('/usr/users/menayat/mae/mae/mae_inference.jpg', bbox_inches='tight')




img_pth = "/scratch/projects/cca/data/livecell/LiveCELL/images/livecell_test_images/MCF7_Phase_H4_1_01d08h00m_4.tif"
img = Image.open(img_pth)

img = img.resize((224, 224))
img = np.array(img)*1.0 #/ 255.
img = img - livecell_mean
img = img / livecell_std
img = np.stack((img,) * 3, axis=-1)


assert img.shape == (224, 224, 3)

# normalize by ImageNet mean and std

print(img.min(), img.max())
plt.rcParams['figure.figsize'] = [5, 5]


chkpt_dir = '/scratch/users/menayat/models/MAE/checkpoint-0.pth'
model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
print('Model loaded.')

torch.manual_seed(2)
print('MAE with pixel reconstruction:')
run_one_image(img, model_mae)
