import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

class _EncoderBlockV2(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, polling=True, bn=False):
        super(_EncoderBlockV2, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels) if bn else nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels) if bn else nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.encode = nn.Sequential(*layers)
        self.pool = None
        if polling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        if self.pool is not None:
            x = self.pool(x)
        return self.encode(x)


class _DecoderBlockV2(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, bn=False):
        super(_DecoderBlockV2, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(middle_channels) if bn else nn.GroupNorm(32, middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels) if bn else nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.decode(x)


class UNetV2(nn.Module):
    def __init__(self, nx, ny , num_classes, in_channels=3, bn=False):
        super(UNetV2, self).__init__()
        en1 = 64
        en2 = 2 * en1; en3 = 2 * en2; en4 = 2 * en3; en5 = 2 * en4; en6 = int(en1/2)
        self.nx = nx
        self.ny = ny
        self.US=nn.Upsample(size=[self.ny-2,self.nx-2],mode='bicubic')
        self.enc1 = _EncoderBlockV2(in_channels, en1, polling=False, bn=bn)
        self.enc2 = _EncoderBlockV2(en1, en2, bn=bn)
        self.enc3 = _EncoderBlockV2(en2, en3, bn=bn)
        self.enc4 = _EncoderBlockV2(en3, en4, bn=bn)
        self.polling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.center = _DecoderBlockV2(en4, en5, en4, bn=bn)
        self.dec4 = _DecoderBlockV2(en5, en4, en3, bn=bn)
        self.dec3 = _DecoderBlockV2(en4, en3, en2, bn=bn)
        self.dec2 = _DecoderBlockV2(en3, en2, en1, bn=bn)
        self.dec1 = nn.Sequential(
            nn.Conv2d(en2, en1, kernel_size=3, padding=1),
            nn.InstanceNorm2d(en1) if bn else nn.GroupNorm(en6, en1),
            nn.ReLU(inplace=True),
            nn.Conv2d(en1, en1, kernel_size=3, padding=1),
            nn.InstanceNorm2d(en1) if bn else nn.GroupNorm(en6, en1),
            nn.ReLU(inplace=True),
        )
        self.conv_8 = nn.Conv2d(en3, num_classes, kernel_size=1)
        self.conv_4 = nn.Conv2d(en2, num_classes, kernel_size=1)
        self.conv_2 = nn.Conv2d(en1, num_classes, kernel_size=1)
        self.final = nn.Conv2d(en1, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        x=self.US(x)
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(self.polling(enc4))
        dec4 = self.dec4(torch.cat([F.interpolate(center, enc4.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc4], 1))
        dec3 = self.dec3(torch.cat([F.interpolate(dec4, enc3.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[-2:], align_corners=False,
                                                  mode='bilinear'), enc1], 1))
        final = self.final(dec1)
        return final


# if __name__ == '__main__':
