import torch
from torch import nn

class BasicConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 upsampling=False,
                 act_norm=False,
                 act_inplace=True):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(*[
                nn.Conv2d(in_channels, out_channels*4, kernel_size=kernel_size,
                          stride=1, padding=padding, dilation=dilation),
                nn.PixelShuffle(2)
            ])
        else:
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation)

        self.norm = nn.GroupNorm(2, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            # trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.act(self.norm(y))
        return y


class ConvSC(nn.Module):

    def __init__(self,
                 C_in,
                 C_out,
                 kernel_size=3,
                 downsampling=False,
                 upsampling=False,
                 act_norm=True,
                 act_inplace=True):
        super(ConvSC, self).__init__()

        stride = 2 if downsampling is True else 1
        padding = (kernel_size - stride + 1) // 2

        self.conv = BasicConv2d(C_in, C_out, kernel_size=kernel_size, stride=stride,
                                upsampling=upsampling, padding=padding,
                                act_norm=act_norm, act_inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        return y


class GroupConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 act_norm=False,
                 act_inplace=True):
        super(GroupConv2d, self).__init__()
        self.act_norm=act_norm
        if in_channels % groups != 0:
            groups=1
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups)
        self.norm = nn.GroupNorm(groups,out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.activate(self.norm(y))
        return y


class gInception_ST(nn.Module):
    """A IncepU block for SimVP"""

    def __init__(self, C_in, C_hid, C_out, incep_ker = [3,5,7,11], groups = 8):        
        super(gInception_ST, self).__init__()
        self.conv1 = nn.Conv2d(C_in, C_hid, kernel_size=1, stride=1, padding=0)

        layers = []
        for ker in incep_ker:
            layers.append(GroupConv2d(
                C_hid, C_out, kernel_size=ker, stride=1,
                padding=ker//2, groups=groups, act_norm=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        y = 0
        for layer in self.layers:
            y += layer(x)
        return y

def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    def __init__(self, channel_in, channel_out, channel_hid, N2, T_out=None, incep_ker=[3,5,7,11], groups=8, **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        self.T_out = T_out

        enc_layers = [gInception_ST(
            channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1,N2-1):
            enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers = [
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups)]
        for i in range(1,N2-1):
            dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_out,
                              incep_ker=incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2-1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1,self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1) )

        if self.T_out is not None:
            T = self.T_out

        y = z.reshape(B, T, C, H, W)
        return y


class SimVP(nn.Module):
    r"""SimVP Model

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, config, **kwargs):
        super(SimVP, self).__init__()
        C, H, W = config["in_varibale_num"], config["h"], config["w"]   # T is pre_seq_length
        self.C_out = config["out_varibale_num"]   # T is pre_seq_length
    
        self.T_in, self.T_out = config["in_times_num"], config["out_times_num"]
        self.residual = config["residual"]
        
        hid_S, hid_T = config["hid_S"], config["hid_T"]
        N_S, N_T = config["N_S"], config["N_T"]
        spatio_kernel_enc, spatio_kernel_dec = config["spatio_kernel_enc"], config["spatio_kernel_dec"]
        
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, self.T_out*self.C_out, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        self.hid = MidIncepNet(self.T_in*hid_S, self.T_out*hid_S, hid_T, N_T, T_out=self.T_out)

        if 0:
            self.skip = nn.Sequential(*[gInception_ST(self.T_in*hid_S, hid_T, hid_T, incep_ker=[3,5,7,11], groups=8),
                                        gInception_ST(hid_T, hid_T, self.T_out*hid_S, incep_ker=[3,5,7,11], groups=8)]) if self.residual else nn.Identity()
        else:
            self.skip = nn.Sequential(*[ConvSC(self.T_in*hid_S, hid_T, downsampling=True),
                                        ConvSC(hid_T, self.T_out*hid_S, upsampling=True)]) if self.residual else nn.Identity()

    def set_loss_func(self, loss_func=None):
        self.loss_func = loss_func

    def forward(self, x_raw, **kwargs):
        base_line = x_raw[:, -1:, :, ...]
        base_line = base_line.repeat(1, self.C_out, 1, 1, 1)  # 初始场作为训练时候的baseline
        B, T, C, H, W = x_raw.shape
        x = x_raw.reshape(B*T, C, H, W)

        embed, skip = self.enc(x)
        _, C_, H_, W_ = embed.shape
        z = embed.view(B, T, C_, H_, W_)
        hid = self.hid(z)
        hid = hid.reshape(B*self.T_out, C_, H_, W_)

        if self.residual:
            skip = skip.view(B, T, C_, H, W)
            skip = skip.reshape(B, -1, H, W)
            skip = self.skip(skip)
            skip = skip.reshape(B*self.T_out, C_, H, W)
        else:
            skip = torch.zeros(B*self.T_out, C_, H, W).to(hid.device)
            
        Y = self.dec(hid, skip)
        Y = Y.reshape(B, self.T_out, self.C_out, H, W)

        return base_line, Y
    
    def predict(self, frames_in, frames_gt=None, compute_loss=False, **kwargs):
        frames_pred = []
        cur_seq = frames_in.clone()
        frames_pred = self.forward(cur_seq)
        
        if compute_loss:
            loss = self.loss_func(frames_pred, frames_gt)
        else:
            loss = None
            
        return frames_pred, loss
    
if __name__ == "__main__":
    config = {
    "hid_S" : 64,
    "hid_T" : 256,
    "N_T": 6,
    "N_S": 2,
    "spatio_kernel_enc": 3,
    "spatio_kernel_dec": 3,
    "in_varibale_num": 1,
    "in_times_num":10,
    "out_varibale_num":1,
    "out_times_num":1,
    "w":308,
    "h":256,
    "residual":False
    }
    model = SimVP(config=config).to('cuda')
    frames_in = torch.randn(2, 10, 1, 308, 256).to('cuda')
    frames_gt = torch.randn(2, 1, 1, 308, 256).to('cuda')
    
    baseline, frames_pred = model(frames_in)
    print(frames_pred.shape)
    
    # Count and print the number of parameters in the model
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Total number of parameters: {total_params}")
    