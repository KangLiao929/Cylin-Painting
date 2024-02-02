import torch.nn as nn
import torch
from torchvision.models import inception_v3

from utils.modules import Flatten, PEConv, Cylin_Block, OutConv
from utils.positional_encodings import PosEncodingSine

class Generator_Cylin(nn.Module):
    def __init__(self, input_dim=4, output_dim=3, embed_dim=32, spe_dim=4, img_w=512, img_h=256):
        super().__init__()
        
        self.layer_spe = PEConv(spe_dim, 1, kernel_size=1, stride=1, padding=0)
        
        self.enc1 = Cylin_Block(input_dim, embed_dim, 5, 1, padding = 0, add_pe=True, pe_down=1, pad_w=512, pad_h=256, pad_ksize=5, pad_stride=1)
        self.enc2 = Cylin_Block(embed_dim, embed_dim*2, 3, 2, 0, add_pe=True, pe_down=2, pad_w=512, pad_h=512, pad_ksize=4, pad_stride=2)
        self.enc3 = Cylin_Block(embed_dim*2, embed_dim*2, 3, 1, 0, add_pe=True, pe_down=2, pad_w=256, pad_h=256, pad_ksize=3, pad_stride=1)
        self.enc4 = Cylin_Block(embed_dim*2, embed_dim*4, 3, 2, 0, add_pe=True, pe_down=4, pad_w=256, pad_h=256, pad_ksize=4, pad_stride=2)
        self.enc5 = Cylin_Block(embed_dim*4, embed_dim*4, 3, 1, 0, add_pe=True, pe_down=4, pad_w=128, pad_h=128, pad_ksize=3, pad_stride=1)
        self.enc6 = Cylin_Block(embed_dim*4, embed_dim*4, 3, 1, 0, add_pe=True, pe_down=4, pad_w=128, pad_h=128, pad_ksize=3, pad_stride=1)
        self.enc7 = Cylin_Block(embed_dim*4, embed_dim*4, 3, 1, 0, dilation=2, add_pe=True, pe_down=4, pad_w=128, pad_h=128, pad_ksize=3, pad_stride=1, pad_dilation=2)
        self.enc8 = Cylin_Block(embed_dim*4, embed_dim*4, 3, 1, 0, dilation=4, add_pe=True, pe_down=4, pad_w=128, pad_h=128, pad_ksize=3, pad_stride=1, pad_dilation=4)
        self.enc9 = Cylin_Block(embed_dim*4, embed_dim*4, 3, 1, 0, dilation=8, add_pe=True, pe_down=4, pad_w=128, pad_h=128, pad_ksize=3, pad_stride=1, pad_dilation=8)
        self.enc10 = Cylin_Block(embed_dim*4, embed_dim*4, 3, 1, 0, dilation=16, add_pe=False, pad_w=128, pad_h=128, pad_ksize=3, pad_stride=1, pad_dilation=16)
        self.enc11 = Cylin_Block(embed_dim*4, embed_dim*4, 3, 1, 0, add_pe=False, pad_w=128, pad_h=128, pad_ksize=3, pad_stride=1)
        
        self.dec1 = Cylin_Block(embed_dim*8, embed_dim*4, 3, 1, 0, add_pe=False, pad_w=128, pad_h=128, pad_ksize=3, pad_stride=1)
        self.dec2 = Cylin_Block(embed_dim*8, embed_dim*2, 3, 1, 0, add_pe=False, pad_w=256, pad_h=256, pad_ksize=3, pad_stride=1)
        self.dec3 = Cylin_Block(embed_dim*4, embed_dim*2, 3, 1, 0, add_pe=False, pad_w=256, pad_h=256, pad_ksize=3, pad_stride=1)
        self.dec4 = Cylin_Block(embed_dim*4, embed_dim, 3, 1, 0, add_pe=False, pad_w=512, pad_h=512, pad_ksize=3, pad_stride=1)
        self.dec5 = Cylin_Block(embed_dim*2, embed_dim, 3, 1, 0, add_pe=False, pad_w=512, pad_h=512, pad_ksize=3, pad_stride=1)
        
        self.output = OutConv(embed_dim, output_dim)

    def forward(self, img, spe):
        
        # learnable SPE
        lpe = self.layer_spe(spe)
        
        # encoder
        e1 = self.enc1(img, lpe)
        e2 = self.enc2(e1, lpe)
        e3 = self.enc3(e2, lpe)
        e4 = self.enc4(e3, lpe)
        e5 = self.enc5(e4, lpe)
        e6 = self.enc6(e5, lpe)
        e7 = self.enc7(e6, lpe)
        e8 = self.enc8(e7, lpe)
        e9 = self.enc9(e8, lpe)
        e10 = self.enc10(e9, lpe)
        e11 = self.enc11(e10, lpe)
        
        # decoder
        d1 = self.dec1(torch.cat([e11, e5], dim=1), lpe)
        d2 = self.dec2(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(torch.cat([d1, e4], dim=1)), lpe)
        d3 = self.dec3(torch.cat([d2, e3], dim=1), lpe)
        d4 = self.dec4(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)(torch.cat([d3, e2], dim=1)), lpe)
        d5 = self.dec5(torch.cat([d4, e1], dim=1), lpe)
        
        # output layer
        out = self.output(d5)
        return out


class Discriminator_Incep(nn.Module):
    def __init__(self):
        super(Discriminator_Incep, self).__init__()

        self.layer1 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(4, 64, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer5 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(),
        )
        self.layer7 = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=0)),
            nn.LeakyReLU(),
        )
        self.layer8 = Flatten()
        self.layer9 = nn.Linear(1280, 256)
        self.layer10 = nn.Linear(1000, 256)
        self.layer11 = nn.Linear(256, 1)

    def forward(self, x, y, z):
        out = torch.cat([x, y], dim=1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out_t = self.layer11(out)

        z = self.layer10(z)
        out = (out * z).sum(1, keepdim=True)
        out = torch.add(out, out_t)
        return out

class InceptionExtractor(nn.Module):
    def __init__(self):
        super(InceptionExtractor, self).__init__()
        self.inception_v3 = inception_v3(pretrained=True,  transform_input=True, aux_logits=False)

    def forward(self, x):
        x = self.inception_v3((x + 1) / 2)
        return x

if __name__ == '__main__':
    x = torch.randn((1, 4, 256, 512))
    pe = PosEncodingSine(24, 256, 512, False)
    pe1 = pe.get(8, 11)
    g = Generator_Cylin(input_dim=4, output_dim=3, embed_dim=32, spe_dim=4)
    y = g(x, pe1.unsqueeze(0))
    print(y.shape)