from SparseViT import SparseViT
import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
from timm.models.layers import trunc_normal_
from decoderhead import Multiple

class SparseViT_Mul(nn.Module):
    def __init__(self, 
                 depth = [5, 8, 20, 7],
                 embed_dim=[64, 128, 320, 512],
                 head_dim=64,
                 img_size=512,
                 s_blocks3=[4, 8, 16, 32],
                 s_blocks4=[8, 16],
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super(SparseViT_Mul, self).__init__()
        self.img_size = img_size
        self.encoder_net = SparseViT(
            layers=depth,
            embed_dim=embed_dim,
            img_size= img_size,
            s_blocks3=s_blocks3,
            s_blocks4=s_blocks4,
            head_dim=head_dim,
            drop_path_rate=0.2,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer
        )
        self.lmu = Multiple(embed_dim=512)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, image, *args, **kwargs):
        image = self.encoder_net(image)
        feature_list = []
        for k, v in image.items():
            feature_list.append(v)
            
        image = self.lmu(feature_list)
        image = F.interpolate(image, size = (self.img_size, self.img_size), mode='bilinear', align_corners=False)
        image = torch.sigmoid(image)
        return image  
    