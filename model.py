import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath
from typing import Type, Callable, Tuple, Optional, Set, List, Union
from utils import A, At

def split_feature(x):
    l = x.shape[1]
    x1 = x[:, 0:l // 2, ::]
    x2 = x[:, l // 2:, ::]
    return x1, x2

class rev_3d_part(nn.Module):

    def __init__(self, in_ch):
        super(rev_3d_part, self).__init__()
        self.f1 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
        )
        self.g1 = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_ch, in_ch, 3, padding=1),
        )

    def forward(self, x):
        x1, x2 = split_feature(x)
        y1 = x1 + self.f1(x2)
        y2 = x2 + self.g1(y1)
        y = torch.cat([y1, y2], dim=1)
        return y

    def reverse(self, y):
        y1, y2 = split_feature(y)
        x2 = y2 - self.g1(y1)
        x1 = y1 - self.f1(x2)
        x = torch.cat([x1, x2], dim=1)
        return x

def window_partition(x, window_size):
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x

def grid_partition(x, grid_size):
    B, D, H, W, C = x.shape
    x = x.view(B, grid_size[0], D // grid_size[0], grid_size[1],  H // grid_size[1], grid_size[2],  W // grid_size[2], C)
    windows = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous().view(-1, reduce(mul, grid_size), C)
    return windows

def grid_reverse(grids, grid_size, B, D, H, W):
    x = grids.view(B, D // grid_size[0], H // grid_size[1], W // grid_size[2], grid_size[0], grid_size[1], grid_size[2], -1)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().view(B, D, H, W, -1)
    return x

class WindowAttention3D(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH


        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N


        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class GridAttention3D(nn.Module):

    def __init__(self, dim, grid_size, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.grid_size = grid_size  # Wd, Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * grid_size[0] - 1) * (2 * grid_size[1] - 1) * (2 * grid_size[2] - 1), num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_d = torch.arange(self.grid_size[0])
        coords_h = torch.arange(self.grid_size[1])
        coords_w = torch.arange(self.grid_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.grid_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.grid_size[1] - 1
        relative_coords[:, :, 2] += self.grid_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.grid_size[1] - 1) * (2 * self.grid_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.grid_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)].reshape(
            N, N, -1)  # Wd*Wh*Ww,Wd*Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wd*Wh*Ww, Wd*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, N, N
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MixBlock3D(nn.Module):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            num_heads: int = 4,
            norm_layer=nn.LayerNorm,
            grid_window_size: Tuple[int, int,int] = (2,7,7),

    ) -> None:
        # Call super constructor
        super(MixBlock3D, self).__init__()
        self.rev_lock = rev_3d_part(128)
        # Init Block and Grid Transformer
        self.norm1 = norm_layer(out_channels)
        self.norm2 = norm_layer(out_channels)
        self.block_transformer = WindowAttention3D(dim=out_channels, window_size=grid_window_size, num_heads=num_heads)
        self.grid_transformer = GridAttention3D(dim=out_channels, grid_size=grid_window_size, num_heads=num_heads)
        self.grid_window_size=grid_window_size
        self.window_partition = window_partition
        self.window_reverse = window_reverse
        self.grid_partition = grid_partition
        self.grid_reverse = grid_reverse
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        B, C, D, H, W = input.shape
        # output = self.mb_conv(input)
        output = rearrange(input, 'b c d h w -> b d h w c')
        output = self.norm1(output)
        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (self.grid_window_size[0] - D % self.grid_window_size[0]) % self.grid_window_size[0]
        pad_b = (self.grid_window_size[1] - H % self.grid_window_size[1]) % self.grid_window_size[1]
        pad_r = (self.grid_window_size[2] - W % self.grid_window_size[2]) % self.grid_window_size[2]
        output = F.pad(output, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, Dp, Hp, Wp, _ = output.shape

        output = self.window_partition(output, self.grid_window_size)
        output = self.block_transformer(output)
        output = self.window_reverse(output, self.grid_window_size, B, Dp, Hp, Wp)
        output_window = output

        output = self.norm2(output)
        output = self.grid_partition(output, self.grid_window_size)
        output = self.grid_transformer(output)
        output = self.grid_reverse(output, self.grid_window_size, B, Dp, Hp, Wp)

        output = output+output_window
        if pad_d1 >0 or pad_r > 0 or pad_b > 0:
            output = output[:, :D, :H, :W, :].contiguous()
        output = rearrange(output, 'b d h w c -> b c d h w')

        output_shortcut1 = input + output

        output = self.rev_lock(output_shortcut1)



        return output

class MixStage(nn.Module):

    def __init__(
            self,
            depth: int,
            in_channels: int,
            out_channels: int,
            num_heads: int = 4,
            grid_window_size: Tuple[int, int] = (2, 7, 7)
    ) -> None:
        super(MixStage, self).__init__()
        self.blocks = nn.Sequential(*[
            MixBlock3D(
                in_channels=in_channels if index == 0 else out_channels,
                out_channels=out_channels,
                num_heads=num_heads,
                grid_window_size=grid_window_size,
            )
            for index in range(depth)
        ])

    def forward(self, input=torch.Tensor) -> torch.Tensor:

        output = self.blocks(input)
        return output

class Mix_For_SCI(nn.Module):

    def __init__(
            self,
            depths: Tuple[int, ...] = (4,4),
            channels: Tuple[int, ...] = (256,256),
            embed_dim: int = 256,
            num_heads: int = 4,
            grid_window_size: Tuple[int, int] = (2, 7, 7),
    ) -> None:

        super(Mix_For_SCI, self).__init__()

        assert len(depths) == len(channels)

        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            # nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(128, 256, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(inplace=True),
        )


        self.stages1 = nn.ModuleList()
        for index, (depth, channel) in enumerate(zip(depths, channels)):
            self.stages1.append(
                MixStage(
                    depth=depth,
                    in_channels=embed_dim if index == 0 else channels[index - 1],
                    out_channels=channel,
                    num_heads=num_heads,
                    grid_window_size=grid_window_size,
                )
            )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                               output_padding=(0, 1, 1)),
            # nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1),
        )


        self.conv3 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            # nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(128, 256, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.stages2 = nn.ModuleList()
        for index, (depth, channel) in enumerate(zip(depths, channels)):
            self.stages2.append(
                MixStage(
                    depth=depth,
                    in_channels=embed_dim if index == 0 else channels[index - 1],
                    out_channels=channel,
                    num_heads=num_heads,
                    grid_window_size=grid_window_size,
                )
            )

        self.conv4 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                               output_padding=(0, 1, 1)),
            # nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1),
        )

        self.conv5 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            # nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(128, 256, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.stages3 = nn.ModuleList()
        for index, (depth, channel) in enumerate(zip(depths, channels)):
            self.stages3.append(
                MixStage(
                    depth=depth,
                    in_channels=embed_dim if index == 0 else channels[index - 1],
                    out_channels=channel,
                    num_heads=num_heads,
                    grid_window_size=grid_window_size,
                )
            )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                               output_padding=(0, 1, 1)),
            # nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1),
        )

        self.convE_UM = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            # nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(128, 256, kernel_size=3, stride=(1, 2, 2), padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.stagesUM = nn.ModuleList()
        for index, (depth, channel) in enumerate(zip(depths, channels)):
            self.stagesUM.append(
                MixStage(
                    depth=depth,
                    in_channels=embed_dim if index == 0 else channels[index - 1],
                    out_channels=channel,
                    num_heads=num_heads,
                    grid_window_size=grid_window_size,
                )
            )

        self.convD1 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                               output_padding=(0, 1, 1)),
            # nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1),
        )

        self.convD2 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                               output_padding=(0, 1, 1)),
            # nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(128, 64, kernel_size=1, stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(64, 1, kernel_size=3, stride=1, padding=1),
        )

        self.convF1 = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 8, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(8, 1, 3, padding=1),
        )
        self.convF2 = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 8, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(8, 1, 3, padding=1),
        )
        self.convF3 = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(16, 8, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(8, 1, 3, padding=1),
        )



    def forward(self, y, Phi, Phi_s):
        '''
        :param y: measurenment
        :param Phi: mask
        :param Phi_s: sum of each frame of Phi
        :return: each stage's output
        '''

        x_list = []
        output = At(y, Phi)

        E_y = torch.div(y, Phi_s)
        E_y = torch.unsqueeze(E_y, dim=1)
        data = E_y.mul(Phi)

        yb = A(output, Phi)
        output = output + At(torch.div(y - yb, Phi_s), Phi)
        output = torch.unsqueeze(output, 1)
        output = torch.cat([output, torch.unsqueeze(data, 1)], dim=1)

        output = self.convE_UM(output)
        for stage in self.stagesUM:
            output = stage(output)
        Recon_output = self.convD1(output)
        UM_output = self.convD2(output)
        Recon_output = torch.squeeze(Recon_output, 1)
        UM_output = torch.squeeze(UM_output, 1)
        x_list.append(Recon_output)
        # x_list.append(UM_output)

        UM_output = torch.unsqueeze(UM_output, 1)
        UM_output1 = self.convF1(UM_output) + UM_output
        UM_output2 = self.convF2(UM_output) + UM_output
        UM_output3 = self.convF3(UM_output) + UM_output

        #####################################################################

        yb = A(output, Phi)
        output = output + At(torch.div(y - yb, Phi_s), Phi)
        output = torch.unsqueeze(output, 1)
        output = torch.cat([output, torch.unsqueeze(data, 1),UM_output1], dim=1)

        output = self.conv1(output)
        for stage in self.stages1:
            output = stage(output)
        output = self.conv2(output)
        output = torch.squeeze(output, 1)
        x_list.append(output)

        yb = A(output, Phi)
        output = output + At(torch.div(y - yb, Phi_s), Phi)
        output = torch.unsqueeze(output, 1)
        output = torch.cat([output, torch.unsqueeze(data, 1), UM_output2], dim=1)

        output = self.conv3(output)
        for stage in self.stages2:
            output = stage(output)
        output_2 = self.conv4(output)
        output_2 = torch.squeeze(output_2, 1)
        x_list.append(output_2)

        yb = A(output_2, Phi)
        output = output_2 + At(torch.div(y - yb, Phi_s), Phi)
        output = torch.unsqueeze(output, 1)
        output = torch.cat([output, torch.unsqueeze(data, 1), UM_output3], dim=1)

        output = self.conv5(output)
        for stage in self.stages3:
            output = stage(output)
        output = self.conv6(output)
        output = torch.squeeze(output, 1)
        x_list.append(output)

        return x_list

