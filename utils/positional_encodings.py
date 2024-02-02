import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class PosEncodingSine():
    def __init__(self, d_model, height, width, pe_vis=False):
        super(PosEncodingSine, self).__init__()
        self.d_model = d_model
        self.height = height
        self.width = width
        self.pe_vis = pe_vis

    def visualization(self, PE):
        c, *_ = PE.shape
        
        fig, axes = plt.subplots(1, c, figsize=(4*c, 4))

        for i in range(c):
            axes[i].imshow(PE[i])
            axes[i].axis('off')
            axes[i].set_title('channel {}'.format(i+1))
        
        plt.tight_layout()
        plt.savefig("PE.jpg")
        plt.show()
        
    def extract(self, PE, i, j):
        
        if(i==j):
            return PE[i].unsqueeze(0)
        else:
            channel_to_extract = list(range(i, j+1))
            selected_channel = [PE[index] for index in channel_to_extract]
            return torch.stack(selected_channel)
    
    def get(self, index1, index2):
        normalize = True
        scale = 2 * math.pi
        temperature = 10000

        d = self.d_model
        h = self.height
        w = self.width

        not_mask = torch.ones(1, h, w)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale
        
        d_half = d // 2 
        dim_t = torch.arange(d_half, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / d_half)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)   
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_x, pos_y), dim=3).permute(0, 3, 1, 2)
        pos = pos.squeeze(0)
        pos = (pos + 1.0) / 2.0
        
        if(self.pe_vis):
            self.visualization(pos)
        
        return self.extract(pos, index1, index2)


if __name__ == '__main__':
    
    pe = PosEncodingSine(24, 256, 512, True)
    # get index 8 to 11 from the all 24-dim spe
    pe1 = pe.get(8, 11)
    