
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean=0.0, std=0.02)


class ConvolutionNorm(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ConvolutionNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, stride=stride, 
                              kernel_size=4, padding=1)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        


    def forward(self, batch, use_norm=True):
        batch = self.conv(batch)
        if use_norm:
            batch = self.instance_norm(batch)
        batch = F.leaky_relu(batch, 0.2)
        return batch

class Discriminator(nn.Module):
    """ PatchGAN discriminator """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = ConvolutionNorm(3, 64)
        self.conv2 = ConvolutionNorm(64, 128)
        self.conv3 = ConvolutionNorm(128, 256)
        self.conv4 = ConvolutionNorm(256, 512, stride=1)
        self.conv_out = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        self.apply(weights_init)
        
    def forward(self, batch):
        batch = self.conv1(batch, use_norm=False)
        batch = self.conv2(batch)
        batch = self.conv3(batch)
        outputs = self.conv4(batch)
        logits = self.conv_out(outputs)
        probs = torch.sigmoid(logits)
        return probs


class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()
        self.refl_padding_1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0)
        self.refl_padding_2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0)
        
    def forward(self, batch):
        padded_batch = self.refl_padding_1(batch)
        conv_batch = self.conv1(padded_batch)
        padded_batch = self.refl_padding_2(conv_batch)
        conv_batch = self.conv2(padded_batch)
        out = batch + conv_batch
        return out

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.refl_padding = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=3, stride=2, padding=0)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        
    def forward(self, batch):
        batch = self.refl_padding(batch)
        batch = self.conv(batch)
        batch = self.instance_norm(batch)
        batch = F.relu(batch)
        return batch

class Generator(nn.Module):
    """ Resnet Generator """
    def __init__(self, num_res=6):
        super(Generator, self).__init__()
        self.num_res = num_res
        # make layers
        self.refl_padding_1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=0)
        self.instance_norm1 = nn.InstanceNorm2d(32)
        self.dk_layer_1 = DownBlock(32, 64)
        self.dk_layer_2 = DownBlock(64, 128)
        self.res_blocks = nn.ModuleList([ResidualBlock(128) for _ in range(num_res)])
        
        self.conv_trans_1 = nn.ConvTranspose2d(128, 64, kernel_size=3, 
                                               stride=2, padding=1, 
                                               output_padding=1)
        self.instance_norm2 = nn.InstanceNorm2d(64)
        self.conv_trans_2 = nn.ConvTranspose2d(64, 32, kernel_size=3, 
                                               stride=2, padding=1, 
                                               output_padding=1)
        self.instance_norm3 = nn.InstanceNorm2d(32)
        self.refl_padding_2 = nn.ReflectionPad2d(3)
        self.conv2 = nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=0)
        # init weights
        self.apply(weights_init)
        
    def forward(self, batch):
        # convolutions
        batch = self.refl_padding_1(batch)
        batch = self.conv1(batch)
        batch = self.instance_norm1(batch)
        batch = F.relu(batch)
        batch = self.dk_layer_1(batch)
        batch = self.dk_layer_2(batch)
        # resconnection blocks
        for i in range(self.num_res):
            batch = self.res_blocks[i](batch)
        # deconvolutions
        batch = self.conv_trans_1(batch)
        batch = self.instance_norm2(batch)
        batch = F.relu(batch)
        batch = self.conv_trans_2(batch)
        batch = self.instance_norm3(batch)
        batch = F.relu(batch)
        # out layers
        batch = self.refl_padding_2(batch)
        batch = self.conv2(batch)
        batch = torch.tanh(batch)
        return batch