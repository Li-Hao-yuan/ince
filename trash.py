import torch
from PIL import Image
import numpy as np
from scipy import linalg

inception_pth = "/root/autodl-tmp/SSDNeRF/work_dirs/cache/inception-2015-12-05.pt"
inception = torch.jit.load(inception_pth).eval().cuda()

img_path1 = "/root/autodl-tmp/SSDNeRF/ssdnerf_gui.png" # gt
img_path2 = "/root/autodl-tmp/SSDNeRF/ssdnerf_gui.png"

img1 = Image.open(img_path1).convert("RGB")
img2 = Image.open(img_path2).convert("RGB")

img1 = torch.from_numpy(np.array(img1)).unsqueeze(0).permute(0, 3, 1, 2).repeat(10,1,1,1)
img2 = torch.from_numpy(np.array(img2)).unsqueeze(0).permute(0, 3, 1, 2).repeat(10,1,1,1)

feature1 = inception(img1.cuda(), return_features=True).cpu().data.numpy()
feature2 = inception(img2.cuda(), return_features=True).cpu().data.numpy()

real_mean = np.mean(feature1, axis=0)
real_cov = np.cov(feature1, rowvar=False)

fake_mean = np.mean(feature2, axis=0)
fake_cov = np.cov(feature2, rowvar=False)

def _calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    """Refer to the implementation from:

    https://github.com/rosinality/stylegan2-pytorch/blob/master/fid.py#L34
    """
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm(
            (sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))
            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(
        real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid, mean_norm, trace

fid, mean_norm, cov = _calc_fid(fake_mean, fake_cov, real_mean, real_cov)
print(fid, mean_norm, cov)