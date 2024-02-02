import argparse
from datetime import datetime
import os
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from data.datasets import *
from model.networks import *
import torch


def train(opt):
    t_start = datetime.now()
    img_path = opt.save_images + opt.method + '/'
    weight_path = opt.save_models + opt.method + '/'
    os.makedirs(img_path, exist_ok=True)
    os.makedirs(weight_path, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)
        device = torch.device('cuda:{}'.format(opt.gpu))
    else:
        device = torch.device('cpu')

    # network define
    generator = Generator_Cylin(input_dim=4, output_dim=3, embed_dim=32, spe_dim=4).to(device)
    incept = InceptionExtractor().to(device)
    discriminator = Discriminator_Incep().to(device)
    
    # Load pretrained models
    if opt.start_epoch != 0:
        generator.load_state_dict(torch.load("saved_models/generator_%d.pth" % opt.start_epoch))
        discriminator.load_state_dict(torch.load("saved_models/discriminator_%d.pth" % opt.start_epoch))

    # network setting
    incept.eval()
    criterion_pixel = torch.nn.L1Loss().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    dataloader = DataLoader(
        ImageDataset(opt.dataset_name, opt.img_w, opt.img_h, opt.spe_dim),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )

    # training
    for epoch in range(opt.start_epoch, opt.n_epochs):
        D_loss = 0
        G_loss = 0
        adv = 0
        pixel = 0
        
        for i, imgs in enumerate(dataloader):
            
            masked_img = Variable(imgs["masked_img"].type(Tensor))
            gt = Variable(imgs["gt"].type(Tensor)) # 
            mask = Variable(imgs["mask"].type(Tensor))
            combo = Variable(imgs['combo'].type(Tensor))
            SPE = Variable(imgs['SPE'].type(Tensor))
            class_cond = incept(gt).detach()

            optimizer_G.zero_grad()
            
            gen_op = generator(combo, SPE)

            # Measure pixel-wise loss against ground truth
            loss_pixel = criterion_pixel(gen_op, gt)
            gen_op_d = gen_op * mask + masked_img

            if epoch < opt.warmup_epochs:
                # Warm-up (pixel-wise loss only)
                loss_pixel.backward()
                optimizer_G.step()
                pixel += loss_pixel.item()
                continue

            # Extract validity predictions from discriminator
            pred_fake = discriminator(gen_op_d, mask, class_cond)

            # Adversarial loss (relativistic average GAN)
            loss_GAN = -pred_fake.mean()

            # Total generator loss
            loss_G = opt.lambda_adv * loss_GAN + loss_pixel

            loss_G.backward()
            optimizer_G.step()

            optimizer_D.zero_grad()

            pred_real = discriminator(gt, mask, class_cond)
            pred_fake = discriminator(gen_op_d.detach(), mask, class_cond)

            # Total loss
            loss_D = nn.ReLU()(1.0 - pred_real).mean() + nn.ReLU()(1.0 + pred_fake).mean()

            loss_D.backward()
            optimizer_D.step()

            D_loss += loss_D.item()
            G_loss += loss_G.item()
            adv += loss_GAN.item()
            pixel += loss_pixel.item()

        avg_D_loss = D_loss / len(dataloader)
        avg_G_loss = G_loss / len(dataloader)
        avg_adv_loss = adv / len(dataloader)
        avg_pixel_loss = pixel / len(dataloader)

        print(
            'Epoch:{1}/{2} D_loss:{3} G_loss:{4} adv:{5} pixel:{6} time:{0}'.format(
                datetime.now() - t_start, epoch + 1, opt.n_epochs, avg_D_loss,
                avg_G_loss, avg_adv_loss, avg_pixel_loss))
        if (epoch + 1) % opt.sample_interval == 0:
            # Save example results
            img_grid = denormalize(torch.cat((masked_img, gen_op, gen_op_d, gt), -1))
            save_image(img_grid, img_path + "epoch-{}.png".format(epoch + 1), nrow=1, normalize=False)
        if (epoch + 1) % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), weight_path + "generator_{}.pth".format(epoch + 1))
            torch.save(discriminator.state_dict(), weight_path + "discriminator_{}.pth".format(epoch + 1))


if __name__ == '__main__':

    print('<==================== setting arguments ===================>\n')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="/dataset/SUN360/train/", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=6, help="size of the batches")
    parser.add_argument("--lr_g", type=float, default=1e-4, help="adam: learning rate of generator")
    parser.add_argument("--lr_d", type=float, default=1e-3, help="adam: learning rate of discriminator")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_w", type=int, default=256, help="img_w")
    parser.add_argument("--img_h", type=int, default=512, help="img_h")
    parser.add_argument("--sample_interval", type=int, default=5, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="batch interval between model checkpoints")
    parser.add_argument("--warmup_epochs", type=int, default=0, help="number of epochs with pixel-wise loss only")
    parser.add_argument("--lambda_adv", type=float, default=1e-2, help="adversarial loss weight")
    parser.add_argument("--save_images", default='/Cylin-Painting/results/', help="where to store images")
    parser.add_argument("--save_models", default='/Cylin-Painting/checkpoints/', help="where to save models")
    parser.add_argument("--method", default='method', help="name of the method")
    parser.add_argument("--gpu", type=int, default=1, help="gpu number")
    parser.add_argument("--spe_dim", type=int, default=24, help="dim of SPE")
    opt = parser.parse_args()
    print(opt)
    
    print('<==================== training ===================>\n')
    
    train(opt)

    