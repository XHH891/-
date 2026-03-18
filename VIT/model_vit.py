import torch
from torch import nn
from EncoderBlock import EncoderBlock

class patch_embedding(nn.Module):
    def __init__(self,patch=16,d=768,in_chans=3,H = 224):
        super(patch_embedding, self).__init__()
        self.patch = patch
        self.d = d
        num_patches = (H // patch) ** 2
        self.fc1 = nn.Linear(in_chans*patch*patch,d)
        self.norm = nn.LayerNorm(d)
        self.cls = nn.Parameter(torch.randn(1,1,d))
        self.pos_embed = nn.Parameter(torch.zeros(1,num_patches+1,d))
    def forward(self, x):
        B = x.shape[0]
        x = self.batch_to_patches(x)
        x = self.fc1(x)
        x = self.norm(x)
        cls_token = self.cls.expand(B,-1,-1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed[:,:x.size(1),:]
        return x
    def batch_to_patches(self,batch_data):
        B, C, H, W = batch_data.shape
        patches = batch_data.unfold(2, self.patch, self.patch)
        patches = patches.unfold(3, self.patch, self.patch)
        patches = patches.permute(0, 2, 3, 1, 4, 5)
        patches = patches.reshape(B, -1, C * self.patch * self.patch)
        return  patches

class vit_encoder(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout,num_layers,use_bias=False,patch=16,d=768,
                 in_chans=3,H = 224, **kwargs):
        super(vit_encoder, self).__init__(**kwargs)
        self.patch = patch_embedding(patch,d,in_chans,H)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(f"{i}",EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
    def forward(self,x):
        x = self.patch(x)
        valid_lens = torch.ones(x.shape[0], device=x.device) * x.shape[1]
        for blk in self.blks:
            x = blk(x,valid_lens)
        return x

class vitmodle(nn.Module):
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, num_layers,class_number,use_bias=False,
                 patch=16,d=768, in_chans=3, H=224,dimension=1024,**kwargs):
        super(vitmodle, self).__init__(**kwargs)
        self.v = vit_encoder( key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, num_layers, use_bias, patch, d, in_chans, H)
        self.fc1 = nn.Linear(d,dimension)
        self.bn1 = nn.BatchNorm1d(dimension)
        self.fc2 = nn.Linear(dimension,class_number)
    def forward(self,x):
        x = self.v(x)
        cls_token = x[:, 0, :]
        a = self.fc1(cls_token)
        a = self.bn1(a)
        a = nn.functional.relu(a)
        a = self.fc2(a)
        return a