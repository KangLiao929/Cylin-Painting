import argparse
import os
from PIL import Image
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ssim
import time

from model.networks import *
from data.datasets import *

def test(opt):
    gt_output = opt.output + 'gt/'
    pre_output = opt.output + 'pre/'
    os.makedirs(pre_output, exist_ok=True)
    os.makedirs(gt_output, exist_ok=True)
    image_path = sorted(glob.glob(opt.dataset_name + "/*.*"))
    img_num = len(image_path)

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
        device = torch.device('cuda:{}'.format(opt.gpu))
    else:
        device = torch.device('cpu')

    generator = Generator_Cylin().to(device)
    generator.load_state_dict(torch.load(opt.model, map_location=torch.device('cpu')))
    generator.eval()
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    dataloader = DataLoader(
        ImageDataset(opt.dataset_name, opt.img_w, opt.img_h, opt.spe_dim),
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )    
    
    avg_ssim = 0
    avg_psnr = 0
    avg_time = 0
    
    for i, imgs in enumerate(dataloader):
            
        masked_img = Variable(imgs["masked_img"].type(Tensor)).to(device)
        gt = Variable(imgs["gt"].type(Tensor)).to(device)
        mask = Variable(imgs["mask"].type(Tensor)).to(device)
        combo = Variable(imgs['combo'].type(Tensor)).to(device)
        SPE = Variable(imgs['SPE'].type(Tensor)).to(device)
    
        with torch.no_grad():
            start_time = time.time()
            gen = generator(combo, SPE)
            end_time = time.time()
            avg_time += end_time - start_time
            fn = image_path[i].split("/")[-1]
            
            gen = gen * mask + masked_img
            gen_de = denormalize_single(gen)
            gt_de = denormalize_single(gt)
            save_image(gen_de, pre_output + fn)
            save_image(gt_de, gt_output + fn)
            
            gen_de = gen_de * 255.
            gt_de = gt_de * 255.
            gen_np = gen_de.squeeze().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
            gt_np = gt_de.squeeze().cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)

            avg_ssim += ssim(gen_np, gt_np, multichannel = True)
            avg_psnr += psnr(gen_np, gt_np)
            
            
            gen_np = Image.fromarray(gen_np.astype(np.uint8))
    
    print("psnr: ", avg_psnr / img_num)
    print("ssim: ", avg_ssim / img_num)
    print("time: ", avg_time / img_num)


if __name__ == '__main__':

    print('<==================== setting arguments ===================>\n')

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="/dataset/360SP/test/", help="name of the dataset")
    parser.add_argument("--output", default='output', help="where to save output")
    parser.add_argument("--model", default="saved_models/generator_10.pth", help="generator model pass")
    parser.add_argument("--img_w", type=int, default=256, help="img_w")
    parser.add_argument("--img_h", type=int, default=512, help="img_h")
    parser.add_argument("--spe_dim", type=int, default=24, help="dim of SPE")
    parser.add_argument("--gpu", type=int, default=3, help="gpu number")
    opt = parser.parse_args()
    print(opt)
    
    print('<==================== testing ===================>\n')
    
    test(opt)