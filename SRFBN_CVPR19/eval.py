import numpy as np
import os
import sys

import matplotlib.pyplot as plt

fin = sys.argv[1]
fout = sys.argv[1]
psnr_full = {}
psnr_patch = {}
ssim_full = {}
ssim_patch = {}

#LOAD THE DATA
imgname = ['baby','bird','butterfly', 'head','woman']
for classname in imgname:
    path1 = os.path.join(fin,'psnr_full_%s_HR_x4.npy' % classname)
    path2 = os.path.join(fin,'psnr_patch_%s_HR_x4.npy' % classname)
    path3 = os.path.join(fin,'ssim_full_%s_HR_x4.npy' % classname)
    path4 = os.path.join(fin,'ssim_patch_%s_HR_x4.npy' % classname)

    psnr_full[classname] = np.load(path1)
    psnr_patch[classname] = np.load(path2)
    ssim_full[classname] = np.load(path3)
    ssim_patch[classname] = np.load(path4)

#GATHER METRICS
for f1 in imgname:
    psnr_diff = psnr_patch[f1] - psnr_full[f1]
    ssim_diff = ssim_patch[f1] - ssim_full[f1]
    out1 = os.path.join(fout, 'psnrdiff_hist_' + f1 + '.png')
    out2 = os.path.join(fout, 'ssimdiff_hist_' + f1 + '.png')
    out3 = os.path.join(fout, 'psnrdiff_' + f1 + '.png')
    out4 = os.path.join(fout, 'ssimdiff_' + f1 + '.png')

    plt.title('PSNR DIFF: ' + f1)
    plt.hist(psnr_diff.flatten(),bins=100)
    plt.savefig(out1)
    plt.clf()

    plt.title('PSNR DIFF: ' + f1)
    plt.imshow(psnr_diff)
    plt.colorbar()
    plt.savefig(out3)
    plt.clf()

    plt.title('SSIM DIFF: ' + f1)
    plt.hist(ssim_diff.flatten(),bins=100)
    plt.savefig(out2)
    plt.clf()

    plt.title('SSIM DIFF: ' + f1)
    plt.imshow(ssim_diff)
    plt.colorbar()
    plt.savefig(out4)
    plt.clf()

