from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import models.utils as utils
import math
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
setup_seed(0)

class CRNNcell(nn.Module):
    """
    Convolutional RNN cell that evolves over both time and iterations.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        dilation
    ):
        """
        Args:
            input_size: Number of input channels
            hidden_size: Number of RCNN hidden layers channels
            kernel_size: Size of convolutional kernel
        """
        super(CRNNcell, self).__init__()
        
        # Convolution for input
        self.i2h = nn.Conv2d(input_size, hidden_size, kernel_size, padding=dilation, dilation=dilation)
        # Convolution for hidden states in temporal dimension
        self.h2h = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=dilation, dilation=dilation)
        # Convolution for hidden states in iteration dimension
        self.ih2ih = nn.Conv2d(hidden_size, hidden_size, kernel_size, padding=dilation, dilation=dilation)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        input: torch.Tensor,
        hidden_iteration: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input: Input 4D tensor of shape `(b, ch, h, w)`
            hidden_iteration: hidden states in iteration dimension, 4d tensor of shape (b, hidden_size, h, w)
            hidden: hidden states in temporal dimension, 4d tensor of shape (b, hidden_size, h, w)
        Returns:
            Output tensor of shape `(b, hidden_size, h, w)`.
        """
        in_to_hid = self.i2h(input)
        hid_to_hid = self.h2h(hidden)
        if hidden_iteration is not None:
            ih_to_ih = self.ih2ih(hidden_iteration)

        if hidden_iteration is not None:
            hidden = self.relu(in_to_hid + hid_to_hid + ih_to_ih)
        else:
            hidden = self.relu(in_to_hid+hid_to_hid)

        return hidden

class BCRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        dilation
    ):
        """
        Args:
            input_size: Number of input channels
            hidden_size: Number of RCNN hidden layers channels
            kernel_size: Size of convolutional kernel
        """
        super(BCRNNlayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.CRNN_model = CRNNcell(input_size, self.hidden_size, kernel_size, dilation)

    def forward(self, input: torch.Tensor, hidden_iteration: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Input 5D tensor of shape `(t, b, ch, h, w)`
            hidden_iteration: hidden states (output of BCRNNlayer) from previous
                    iteration, 5d tensor of shape (t, b, hidden_size, h, w)
        Returns:
            Output tensor of shape `(t, b, hidden_size, h, w)`.
        """
        t, b, ch, h, w = input.shape
        size_h = [b, self.hidden_size, h, w]
        
        # hid_init = Variable(torch.zeros(size_h)).to(input.device)
        hid_init = torch.Tensor(torch.zeros(size_h)).requires_grad_(True).to(input.device)
        output_f = []
        output_b = []
        
        # forward
        hidden = hid_init
        for i in range(t):
            if hidden_iteration is not None:
                hidden = self.CRNN_model(input[i], hidden_iteration[i], hidden)
            else:
                hidden = self.CRNN_model(input[i], hidden_iteration, hidden)
            output_f.append(hidden)
        output_f = torch.cat(output_f)

        # backward
        hidden = hid_init
        for i in range(t):
            if hidden_iteration is not None:
                hidden = self.CRNN_model(input[t - i - 1], hidden_iteration[t - i -1], hidden)
            else:
                hidden = self.CRNN_model(input[t - i - 1], hidden_iteration, hidden)
            output_b.append(hidden)
        output_b = torch.cat(output_b[::-1])  # t b hidden_size h w

        output = output_f + output_b

        if b == 1:
            output = output.view(t, 1, self.hidden_size, h, w)
        else:
            output = output.view(t, -1, self.hidden_size, h, w)
        return output

class CRNNlayer(nn.Module):
    """
    Bidirectional Convolutional RNN layer
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        kernel_size: int,
        dilation,
        left=True
    ):
        """
        Args:
            input_size: Number of input channels
            hidden_size: Number of RCNN hidden layers channels
            kernel_size: Size of convolutional kernel
        """
        super(CRNNlayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.left = left
        self.CRNN_model = CRNNcell(input_size, self.hidden_size, kernel_size, dilation)

    def forward(self, input: torch.Tensor, hidden_iteration: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Input 5D tensor of shape `(t, b, ch, h, w)`
            hidden_iteration: hidden states (output of BCRNNlayer) from previous
                    iteration, 5d tensor of shape (t, b, hidden_size, h, w)
        Returns:
            Output tensor of shape `(t, b, hidden_size, h, w)`.
        """
        t, b, ch, h, w = input.shape
        size_h = [b, self.hidden_size, h, w]
        
        # hid_init = Variable(torch.zeros(size_h)).to(input.device)
        hid_init = torch.Tensor(torch.zeros(size_h)).requires_grad_(True).to(input.device)
        output_f = []
        output_b = []
        
        # forward
        hidden = hid_init
        for i in range(t):
            if self.left:
                hidden = self.CRNN_model(input[i], hidden_iteration[i], hidden)
                output_f.append(hidden)
                output = torch.cat(output_f)
            else:
                hidden = self.CRNN_model(input[t - i - 1], hidden_iteration[t - i -1], hidden)
                output_f.append(hidden)
                output = torch.cat(output_f[::-1])

        if b == 1:
            output = output.view(t, 1, self.hidden_size, h, w)
        else:
            output = output.view(t, -1, self.hidden_size, h, w)
        return output

class Conv2_1d(nn.Module):
    def __init__(self, in_chans, chans, kernel_size=(3, 2, 2), stride=(1, 2, 2),
                        padding=(0, 0, 0), dilation=(1, 1, 1), bias=True, tcp=True):
        super().__init__()
        self.tcp = tcp
        self.conv2d = nn.Conv2d(in_chans, chans, kernel_size[1:], stride[1:], padding[1:], dilation=dilation[1:], bias=bias)
        if self.tcp:
            self.conv1d = nn.Conv1d(chans, chans, kernel_size[0], stride[0], padding[0], dilation=dilation[0], bias=bias)
        else:
            self.conv1d = nn.Conv1d(chans, chans, kernel_size[0], stride[0], 1, dilation=dilation[0], bias=bias)

    def forward(self, x):
        #2D convolution
        b=1
        t, c, d1, d2 = x.size()
        x = F.relu(self.conv2d(x))
        
        #1D convolution
        
        c, dr1, dr2 = x.size(1), x.size(2), x.size(3)
        x = x.view(b, t, c, dr1, dr2)
        x = x.permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(b*dr1*dr2, c, t)
        if self.tcp:
            x = torch.cat((x[:,:,t-1:t], x, x[:,:,0:1]), dim=-1)
        x = self.conv1d(x)

        #Final output
        out_c, out_t = x.size(1), x.size(2)
        x = x.view(b, dr1, dr2, out_c, out_t)
        x = x.permute(0, 4, 3, 1, 2).contiguous().reshape(-1, out_c, dr1, dr2)
        return x

class ConvTranspose2_1d(nn.Module):
    def __init__(self, in_chans, chans, kernel_size=(3, 2, 2), stride=(1, 2, 2),
                        padding=(0, 0, 0), dilation=(1, 1, 1), bias=True, tcp=True):
        super().__init__()
        self.tcp = tcp
        self.convTranspose2d = nn.ConvTranspose2d(in_chans, chans, kernel_size[1:], stride[1:], padding[1:], dilation=dilation[1:], bias=bias)
        # self.convTranspose2d = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        #         nn.Conv2d(in_chans, chans, 3, stride=1, padding=1, bias=False))
        if self.tcp:
            self.convTranspose1d = nn.Conv1d(chans, chans, kernel_size[0], stride[0], padding[0], dilation=dilation[0], bias=bias)
        else:
            self.convTranspose1d = nn.Conv1d(chans, chans, kernel_size[0], stride[0], 1, dilation=dilation[0], bias=bias)
    
    def forward(self, x):
        b=1
        t, c, d1, d2 = x.size()
        x = x.view(b*t, c, d1, d2)
        x = F.relu(self.convTranspose2d(x))
        
        c, dr1, dr2 = x.size(1), x.size(2), x.size(3)
        x = x.view(b, t, c, dr1, dr2)
        x = x.permute(0, 3, 4, 2, 1).contiguous()
        x = x.view(b*dr1*dr2, c, t)
        if self.tcp:
            x = torch.cat((x[:,:,t-1:t], x, x[:,:,0:1]), dim=-1)
        x = self.convTranspose1d(x)

        #Final output
        out_c, out_t = x.size(1), x.size(2)
        x = x.view(b, dr1, dr2, out_c, out_t)
        x = x.permute(0, 4, 3, 1, 2).contiguous().reshape(-1, out_c, dr1, dr2)
        return x


class CRUNet_D_Block(nn.Module):
    def __init__(self, chans=64, norms=True, tcp=True):
        super().__init__()
        self.chans = chans
        self.chans1 = chans
        self.norms = norms
        crnn_op = CRNNlayer
        
        self.d_conv = Conv2_1d(self.chans, self.chans, tcp=tcp)
        self.d_crnn1 = crnn_op(input_size=self.chans, hidden_size=self.chans, kernel_size=3, dilation=1) # previous 1 2 4 2 1 
        self.d_crnn2 = crnn_op(input_size=self.chans1, hidden_size=self.chans1, kernel_size=3, dilation=2)
        
        self.bcrnn = BCRNNlayer(input_size=self.chans1, hidden_size=self.chans1, kernel_size=3, dilation=4)

        self.u_crnn1 = crnn_op(input_size=self.chans1, hidden_size=self.chans1, kernel_size=3, dilation=2, left=False)
        self.u_crnn2 = crnn_op(input_size=self.chans, hidden_size=self.chans, kernel_size=3, dilation=1, left=False)
        
        self.conv11 = nn.Conv2d(self.chans, self.chans, 1)
        self.conv21 = nn.Conv2d(self.chans1, self.chans1, 1)
        self.conv31 = nn.Conv2d(self.chans1, self.chans1, 1)
        self.conv41 = nn.Conv2d(self.chans1, self.chans1, 1)
        self.conv51 = nn.Conv2d(self.chans1, self.chans1, 1)
        
        self.u_conv = ConvTranspose2_1d(self.chans1, self.chans, tcp=tcp)
        self.convs1 = Conv2_1d(self.chans1, self.chans1, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,2,2), dilation=(1,2,2), tcp=tcp)
        self.convs2 = Conv2_1d(self.chans, self.chans, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,4,4), dilation=(1,4,4), tcp=tcp)

        self.conv1 = Conv2_1d(self.chans1*2, self.chans1, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,2,2), dilation=(1,2,2), tcp=tcp)
        self.conv2 = Conv2_1d(self.chans*2, self.chans, kernel_size=(3,3,3), stride=(1,1,1), padding=(0,1,1), tcp=tcp)
        
        self.conv = nn.Conv2d(self.chans, 2, 5, padding = 2)
        self.conv0 = nn.Conv2d(2, self.chans, 3, padding = 1)

        self.relu = nn.LeakyReLU(inplace=True)
        self.pp = nn.Parameter(torch.zeros(1, 2, 224, 112), requires_grad=True)
    
    def sens_expand(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Forward operator: from coil-combined image-space to k-space.
        """
        return utils.fft2c(utils.complex_mul(x, sens_maps))

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Backward operator: from k-space to coil-combined image-space.
        """
        x = utils.ifft2c(x)
        return utils.complex_mul(x, utils.complex_conj(sens_maps)).sum(
            dim=2, keepdim=True,
        )
        
    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        x = x.reshape(b, c * h * w)

        mean = x.mean(dim=1).view(b, 1, 1, 1)
        std = x.std(dim=1).view(b, 1, 1, 1)

        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std

    def unnorm(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        return x * std + mean
    
    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 7) + 1
        h_mult = ((h - 1) | 7) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ) -> torch.Tensor:
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def data_consistency(self, img, k0, mask, sens_maps, noise_lvl=None):
        v = noise_lvl
        k = torch.view_as_complex(self.sens_expand(img, sens_maps))
        if v is not None:  # noisy case
            out = (1 - mask) * k + mask * (k + v * k0) / (1 + v)
        else:  # noiseless case
            # mask = mask.unsqueeze(2)
            out = (1 - mask) * k + mask * k0

        out = self.sens_reduce(torch.view_as_real(out), sens_maps).squeeze(2)  # b t cm h w 2
        return out
    
    def cal_std_map(self, x, x0):
        _, _, h, w = x.shape
        x0 = utils.complex_abs(x0.permute(0,2,3,1).contiguous()).unsqueeze(1)
        x_std = torch.std(x0, dim=0, keepdim=True)
        x_std = x_std/(torch.norm(x_std)+1e-6)
        ones_tensor = F.interpolate(self.pp, (h, w), mode="bilinear")
        x_std = x_std * ones_tensor
        x_std.requires_grad_(True)
        return x_std*x+x
    
    def forward(self, ref_kspace, x, net, mask, sens_maps): 
        b, ch, h, w, t = x.size()       
        indices = list(range(12))
        if self.norms:
            x = x.permute(0,1,4,2,3).contiguous()
            x = x.reshape(b, -1, h, w)
            x, mean, std = self.norm(x)
            x = x.reshape(b, 2, -1, h, w)      
            x = x.permute(2,0,1,3,4).contiguous() # t b ch h w
            x = x.float().contiguous()
        else:
            x = x.permute(4, 0, 1, 2, 3).contiguous()
            x = x.float()
        x = x.reshape(-1, 2, h, w)
        x, pad_sizes = self.pad(x)
        _, _, h, w = x.shape

        x0 = self.conv0(x)
        
        net['x0'] = net['x0'].view(t, b, self.chans, h, w)
        net['x0'] = self.d_crnn1(x0.reshape(t, b, self.chans, h, w), net['x0'])
        net['x0'] = net['x0'].reshape(-1, self.chans, h, w)+self.conv11(x0)
        x01 = self.d_conv(net['x0'])

        net['x1'] = net['x1'].view(t, b, self.chans1, h//2, w//2)
        net['x1'] = self.d_crnn2(x01.reshape(t, b, self.chans1, h//2, w//2), net['x1'])+self.conv21(x01).reshape(t, b, self.chans1, h//2, w//2)
        net['x1'] = self.convs1(net['x1'].reshape(-1, self.chans, h//2, w//2))
        
        net['x2'] = net['x2'].view(t, b, self.chans1, h//2, w//2)
        net['x2'] = self.bcrnn(net['x1'].reshape(t, b, self.chans1, h//2, w//2), net['x2'])+self.conv31(net['x1'].reshape(-1, self.chans1, h//2, w//2)).reshape(t, b, self.chans1, h//2, w//2)
        net['x2'] = self.convs2(net['x2'].reshape(-1, self.chans, h//2, w//2))

        net['x3'] = net['x3'].view(t, b, self.chans1, h//2, w//2)
        net['x3'] = self.u_crnn1(net['x2'].reshape(t, b, self.chans1, h//2, w//2), net['x3'])+self.conv41(net['x2'].reshape(-1, self.chans1, h//2, w//2)).reshape(t, b, self.chans1, h//2, w//2)
        net['x3'] = torch.cat((net['x1'].reshape(-1, self.chans1, h//2, w//2), net['x3'].reshape(-1, self.chans1, h//2, w//2)), dim=1)
        net['x3'] = self.conv1(net['x3'])
        net['x1'] = net['x3']
        
        x01 = self.u_conv(net['x3'])
        
        net['x4'] = net['x4'].view(t, b, self.chans, h, w)
        net['x4'] = self.u_crnn2(x01.reshape(t, b, self.chans, h, w), net['x4'])+self.conv51(x01).reshape(t, b, self.chans, h, w)
        net['x4'] = torch.cat((net['x0'].reshape(-1, self.chans, h, w), net['x4'].reshape(-1, self.chans, h, w)), dim=1)
        net['x4'] = self.conv2(net['x4'])
        net['x0'] = net['x4']
        
        x01 = self.conv(net['x4'])
        x = x + x01 # tb 2 h w
        # x = x + self.cal_std_map(x01, x)
        x = self.unpad(x, *pad_sizes)
        _, _, h, w = x.shape

        if self.norms:
            x = x.view(-1, b, ch, h, w) # t b 2 h w
            x = x.permute(1,0,2,3,4).contiguous()
            x = x.reshape(b, -1, h, w)
            x = self.unnorm(x, mean, std)
            x = x.reshape(b,-1, 2, h, w)
            x = x.permute(0,1,3,4,2).contiguous()
        else:
            x = x.view(-1, b, ch, h, w)
            x = x.permute(1,0,3,4,2).contiguous()
        
        x = self.data_consistency(x.unsqueeze(2), ref_kspace, mask, sens_maps)
        x = x.permute(0,4,2,3,1).contiguous() # b 2 h w t

        return x, net


class CRUNet_D_NWS(nn.Module):
    def __init__(self, num_cascades=5, chans=64):
        super().__init__()
        self.num_cascades = num_cascades
        self.chans = chans
        self.chans1 = chans
        self.cascades = nn.ModuleList(
            [CRUNet_D_Block(chans=chans) for _ in range(self.num_cascades)]
        )

    def sens_reduce(self, x: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        """
        Backward operator: from k-space to coil-combined image-space.
        """
        x = utils.ifft2c(x)
        return utils.complex_mul(x, utils.complex_conj(sens_maps)).sum(
            dim=2, keepdim=True,
        )
    
        
    def pad(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        _, _, h, w = x.shape
        w_mult = ((w - 1) | 7) + 1
        h_mult = ((h - 1) | 7) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]
        x = F.pad(x, w_pad + h_pad)

        return x, (h_pad, w_pad, h_mult, w_mult)
    
    def forward(self, ref_kspace, mask, sens_maps):
        x_ref = self.sens_reduce(torch.view_as_real(ref_kspace), sens_maps).squeeze(2) # b t h w 2
        x = x_ref.clone().permute(0,4,2,3,1).contiguous() # b 2 h w t
        b, ch, h, w, t = x.size()

        net = {}
        rcnn_layers = 6
        net['x0'] = torch.Tensor(torch.zeros([t*b, self.chans, h, w])).requires_grad_(True).to(x.device)
        net['x0'], _ = self.pad(net['x0'])
        net['x4'] = torch.Tensor(torch.zeros([t*b, self.chans, h, w])).requires_grad_(True).to(x.device)
        net['x4'], _ = self.pad(net['x4'])
        tb, _, hh, ww = net['x0'].shape
        size_h = [tb, self.chans1, hh//2, ww//2]
        for j in range(1, rcnn_layers-2):
            net['x%d'%j] = torch.Tensor(torch.zeros(size_h)).requires_grad_(True).to(x.device)
        
        for cascade in self.cascades:
            x, net = cascade(ref_kspace, x, net, mask, sens_maps)
        return x.permute(0,4,2,3,1).contiguous()
