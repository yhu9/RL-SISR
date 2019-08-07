import argparse, time, os
import imageio

import torch
import numpy as np
import cv2

import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset


def main():
    parser = argparse.ArgumentParser(description='Test Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt)
    opt = option.dict_to_nonedict(opt)

    # initial configure
    scale = opt['scale']
    degrad = opt['degradation']
    network_opt = opt['networks']
    model_name = network_opt['which_model'].upper()
    if opt['self_ensemble']: model_name += 'plus'

    # create test dataloader
    bm_names =[]
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)
        print('===> Test Dataset: [%s]   Number of images: [%d]' % (test_set.name(), len(test_set)))
        bm_names.append(test_set.name())

    # create solver (and load model)
    solver = create_solver(opt)

    # Test phase
    print('===> Start Test')
    print("==================================================")
    print("Method: %s || Scale: %d || Degradation: %s"%(model_name, scale, degrad))

    for bm, test_loader in zip(bm_names, test_loaders):
        print("Test set : [%s]"%bm)

        sr_list = []
        path_list = []

        total_psnr = []
        total_ssim = []
        total_time = []

        need_HR = False if test_loader.dataset.__class__.__name__.find('LRHR') < 0 else True

        step = 4
        for iter, batch in enumerate(test_loader):
            b,d,h,w = batch['LR'].shape

            if h % step != 0:
                hcut = h % step
                batch['LR'] = batch['LR'][:,:,:h-hcut,:]
                batch['HR'] = batch['HR'][:,:,:-hcut*4,:]
            if w % step != 0:
                wcut = w % step
                batch['LR'] = batch['LR'][:,:,:,:w-wcut]
                batch['HR'] = batch['HR'][:,:,:,:-wcut*4]
            b,d,h,w = batch['LR'].shape

            solver.feed_data(batch,need_HR=need_HR)
            t0 = time.time()
            solver.test()
            t1 = time.time()
            visuals = solver.get_current_visual(need_HR=need_HR)
            SR_img = visuals['SR']
            HR_img = visuals['HR']
            normal_sr_path = 'full_' + os.path.basename(batch['HR_path'][0])[:-4]

            cv2.imshow('FULL SR IMG', (cv2.cvtColor(SR_img,cv2.COLOR_BGR2RGB)).astype(np.uint8))

            sr_list.append(SR_img)
            path_list.append(os.path.join('out','SR_' + normal_sr_path + '.png'))
            # calculate PSNR/SSIM metrics on Python
            if need_HR:
                psnr, ssim = util.calc_metrics(SR_img, HR_img, crop_border=scale,out=normal_sr_path)
                total_psnr.append(psnr)
                total_ssim.append(ssim)
                print("[%d/%d] %s || PSNR(dB)/SSIM: %.2f/%.4f || Timer: %.4f sec ." % (iter+1, len(test_loader),
                                                                                       os.path.basename(batch['LR_path'][0]),
                                                                                       psnr, ssim,
                                                                                       (t1 - t0)))
            else:
                print("[%d/%d] %s || Timer: %.4f sec ." % (iter + 1, len(test_loader),
                                                           os.path.basename(batch['LR_path'][0]),
                                                           (t1 - t0)))

            canvas = np.zeros((h*4,w*4,3))
            for i in range(0,h-1,step):
                for j in range(0,w-1,step):
                    mybatch = {}
                    mybatch['LR_path'] = batch['LR_path']
                    mybatch['HR_path'] = batch['HR_path']
                    mybatch['LR'] = batch['LR'][:,:,i:i+step,j:j+step]
                    mybatch['HR'] = batch['HR'][:,:,i*4:(i+step)*4,j*4:(j+step)*4]

                    # calculate forward time
                    solver.feed_data(mybatch, need_HR=need_HR)
                    t0 = time.time()
                    solver.test()
                    t1 = time.time()
                    total_time.append((t1 - t0))

                    visuals = solver.get_current_visual(need_HR=need_HR)
                    canvas[i*4:(i+step)*4,j*4:(j+step)*4,:] = visuals['SR']

            cv2.imshow('PATCH SR IMG', cv2.cvtColor(canvas.astype(np.uint8),cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)

            # calculate PSNR/SSIM metrics on Python
            patch_sr_path = 'patch_' + os.path.basename(batch['HR_path'][0])[:-4]
            sr_list.append(canvas)
            path_list.append(os.path.join('out','SR_' + patch_sr_path + '.png'))
            if need_HR:
                psnr, ssim = util.calc_metrics(canvas, HR_img, crop_border=scale,out=patch_sr_path)
                total_psnr.append(psnr)
                total_ssim.append(ssim)
                print("[%d/%d] %s || PSNR(dB)/SSIM: %.2f/%.4f || Timer: %.4f sec ." % (iter+1, len(test_loader),
                                                                                       os.path.basename(batch['LR_path'][0]),
                                                                                       psnr, ssim,
                                                                                       (t1 - t0)))
            else:
                print("[%d/%d] %s || Timer: %.4f sec ." % (iter + 1, len(test_loader),
                                                   os.path.basename(batch['LR_path'][0]),
                                                   (t1 - t0)))

        if need_HR:
            print("---- Average PSNR(dB) /SSIM /Speed(s) for [%s] ----" % bm)
            print("PSNR: %.2f      SSIM: %.4f      Speed: %.4f" % (sum(total_psnr)/len(total_psnr),
                                                                  sum(total_ssim)/len(total_ssim),
                                                                  sum(total_time)/len(total_time)))
        else:
            print("---- Average Speed(s) for [%s] is %.4f sec ----" % (bm,
                                                                      sum(total_time)/len(total_time)))

        # save SR results for further evaluation on MATLAB
        if need_HR:
            save_img_path = os.path.join('./results/SR/'+degrad, model_name, bm, "x%d"%scale)
        else:
            save_img_path = os.path.join('./results/SR/'+bm, model_name, "x%d"%scale)

        print("===> Saving SR images of [%s]... Save Path: [%s]\n" % (bm, save_img_path))

        if not os.path.exists(save_img_path): os.makedirs(save_img_path)
        for img, name in zip(sr_list, path_list):
            imageio.imwrite(name, img)

    print("==================================================")
    print("===> Finished !")

if __name__ == '__main__':
    with torch.cuda.device(2):
        main()
