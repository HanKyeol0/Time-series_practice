import torch
import torch.nn as nn


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
    
class conv_resize(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_adjust = 2, init_weight=True):
        super(conv_resize, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = 2 * kernel_adjust - 1
        padding = (kernel_size - 1) // 2
        self.middlefeature_conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, 
                                           kernel_size = kernel_size, padding = padding)
        self.lowfeature_conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size)
        self.highfeature_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) :
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        x_high, x_middle = self.highfeature_conv(x), self.middlefeature_conv(x)
        return x_high, x_middle
    
class conv_resizeback(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_adjust = 2, init_weight=True):
        super(conv_resizeback, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = 2 * kernel_adjust - 1
        padding = (kernel_size - 1) // 2
        self.middlefeature_conv = nn.Conv2d(in_channels = self.in_channels, out_channels = self.out_channels, 
                                           kernel_size = kernel_size, padding = padding)
        self.highfeature_conv = nn.Conv2d(in_channels = self.in_channels, out_channels = self.out_channels,
                                           kernel_size = kernel_size)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, high_enc_out, middle_enc_out):
        x_high, x_middle = self.highfeature_conv(high_enc_out), self.middlefeature_conv(middle_enc_out)
        x = (x_high + x_middle)/2
        return x


class conv_resize_up_scailing(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_adjust = 2, num_kernels = 3, init_weight=True):
        super(conv_resize_up_scailing, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(1, self.num_kernels + 1):
            kernels.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size= 2 * i - 1, stride = i, padding = (2 * i - 1) // 2))
        self.kernels = nn.ModuleList(kernels)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) :
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        return res_list

class conv_resizeback_up_scailing(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_adjust = 2, num_kernels = 3, init_weight=True):
        super(conv_resizeback_up_scailing, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(1, self.num_kernels + 1):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size= 2 * i - 1, stride = i, padding = (2 * i - 1) // 2))
        self.kernels = nn.ModuleList(kernels)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, conv_masked_output):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](conv_masked_output[i]))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class MultiScaleAugmentation(nn.Module):
    def __init__(self, d_model, num_kernels = 2, init_weight=True):
        super(MultiScaleAugmentation, self).__init__()
        self.in_channels = d_model
        self.out_channels = d_model
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList()
        
        for i in range(num_kernels,0,-1):
            self.kernels.append(nn.Conv2d(self.in_channels, self.out_channels, 
                                           kernel_size=2 * i + 1, padding=i))

        self.kernels.append(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0))

        for i in range(1, self.num_kernels+1):
            self.kernels.append(nn.ConvTranspose2d(self.in_channels, self.out_channels, 
                                                 kernel_size= 2 * i + 1, stride = i + 1, padding = i))

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) :
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        res_list = []
        for i in range(len(self.kernels)):
            res_list.append(self.kernels[i](x))
        return res_list

class MultiScaleAugmentationForTime(nn.Module):
    def __init__(self, d_model, num_kernels = 2, init_weight=True):
        super(MultiScaleAugmentationForTime, self).__init__()
        self.in_channels = d_model
        self.out_channels = d_model
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList()
        
        for i in range(num_kernels,0,-1):
            self.kernels.append(nn.Conv2d(self.in_channels, self.out_channels, 
                                           kernel_size=2 * i + 1, padding=i))

        self.kernels.append(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0))

        for i in range(1, self.num_kernels+1):
            self.kernels.append(nn.ConvTranspose2d(self.in_channels, self.out_channels, 
                                                 kernel_size= 2 * i + 1, stride = i + 1, padding = i))

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) :
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        res_list = []
        for i in range(len(self.kernels)):
            breakpoint()
            res_list.append(self.kernels[i](x))
        return res_list

class MultiScaleAugmentationBack(nn.Module):
    def __init__(self, d_model, num_kernels = 2, init_weight=True):
        super(MultiScaleAugmentationBack, self).__init__()
        self.in_channels = d_model
        self.out_channels = d_model
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList()

        for i in range(num_kernels,0,-1):
            self.kernels.append(nn.Conv2d(self.in_channels, self.out_channels, 
                                           kernel_size=2 * i + 1, padding=i))

        self.kernels.append(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0))
        
        for i in range(1, self.num_kernels+1):
            self.kernels.append(nn.Conv2d(self.in_channels, self.out_channels, 
                                                 kernel_size= 2 * i - 1,
                                                #  stride = i,
                                                 padding = (2 * i - 1) // 2))
        if init_weight:
            self._initialize_weights()
            
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) :
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        res_list = []
        breakpoint()
        for i in range(len(self.kernels)):
            res_list.append(self.kernels[i](x[i]))
        return res_list    

# class MultiScaleAugmentationBack(nn.Module):
#     def __init__(self, d_model, num_kernels = 2, init_weight=True):
#         super(MultiScaleAugmentationBack, self).__init__()
#         self.in_channels = d_model
#         self.out_channels = d_model
#         self.num_kernels = num_kernels
#         self.kernels = nn.ModuleList()

#         for i in range(num_kernels,0,-1):
#             self.kernels.append(nn.Conv2d(self.in_channels, self.out_channels, 
#                                            kernel_size=2 * i + 1, padding=i))

#         self.kernels.append(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0))
        
#         for i in range(1, self.num_kernels+1):
#             self.kernels.append(nn.Conv2d(self.in_channels, self.out_channels, 
#                                                  kernel_size= 2 * i - 1,
#                                                 #  stride = i,
#                                                  padding = (2 * i - 1) // 2))
#         if init_weight:
#             self._initialize_weights()
            
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.ConvTranspose2d) :
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
                    
#     def forward(self, x):
#         res_list = []
#         for i in range(len(self.kernels)):
#             res_list.append(self.kernels[i](x[i]))
#         return res_list
    
class MultiScaleAugmentation2(nn.Module):
    def __init__(self, channels, x_mark_channels=4,num_kernels = 2, init_weight=True):
        super(MultiScaleAugmentation2, self).__init__()
        self.in_channels = channels
        self.x_mark_in_channel = x_mark_channels
        self.out_channels = channels
        self.num_kernels = num_kernels
        self.x_kernels = nn.ModuleList()
        self.xmark_kernels = nn.ModuleList()
        
        # x rescale
        for i in range(num_kernels,0,-1):
            self.x_kernels.append(nn.Conv2d(self.in_channels, self.out_channels, 
                                           kernel_size=2 * i + 1, padding=i))

        self.x_kernels.append(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0))

        for i in range(1, self.num_kernels+1):
            self.x_kernels.append(nn.ConvTranspose2d(self.in_channels, self.out_channels, 
                                                 kernel_size= 2 * i + 1, stride = i + 1, padding = i))
            
        # x_mark rescale
        for i in range(num_kernels,0,-1):
            self.xmark_kernels.append(nn.Conv2d(self.x_mark_in_channel, self.out_channels, 
                                           kernel_size=2 * i + 1, padding=i))

        self.xmark_kernels.append(nn.Conv2d(self.x_mark_in_channel, self.out_channels, kernel_size=1, padding=0))

        for i in range(1, self.num_kernels+1):
            self.xmark_kernels.append(nn.ConvTranspose2d(self.x_mark_in_channel, self.out_channels, 
                                                 kernel_size= 2 * i + 1, stride = i + 1, padding = i))

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) :
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x, x_mark):
        res_x = []
        res_x_mark = []
        for i in range(len(self.x_kernels)):
            res_x.append(self.x_kernels[i](x))
            res_x_mark.append(self.xmark_kernels[i](x_mark))
        return res_x, res_x_mark
    
    
class MultiScaleAugmentationBack2(nn.Module):
    def __init__(self, d_model, num_kernels = 2, init_weight=True):
        super(MultiScaleAugmentationBack, self).__init__()
        self.in_channels = d_model
        self.out_channels = d_model
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList()

        for i in range(num_kernels,0,-1):
            self.kernels.append(nn.Conv2d(self.in_channels, self.out_channels, 
                                           kernel_size=2 * i + 1, padding=i))

        self.kernels.append(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0))
        
        for i in range(1, self.num_kernels+1):
            self.kernels.append(nn.Conv2d(self.in_channels, self.out_channels, 
                                                 kernel_size= 2 * i + 1))
        
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) :
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        res_list = []
        for i in range(len(self.kernels)):
            res_list.append(self.kernels[i](x[i]))
        return res_list
    
class MultiScaleBlock(nn.Module):
    def __init__(self, d_model, num_kernels = 2, init_weight=True):
        super(MultiScaleAugmentation, self).__init__()
        self.in_channels = d_model
        self.out_channels = d_model
        self.num_kernels = num_kernels
        self.kernels = nn.ModuleList()
        
        for i in range(num_kernels,0,-1):
            self.kernels.append(nn.Conv2d(self.in_channels, self.out_channels, 
                                           kernel_size=2 * i + 1, padding=i))

        self.kernels.append(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, padding=0))

        for i in range(1, self.num_kernels+1):
            self.kernels.append(nn.ConvTranspose2d(self.in_channels, self.out_channels, 
                                                 kernel_size= 2 * i + 1, stride = i + 1, padding = i))

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) :
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        res_list = []
        for i in range(len(self.kernels)):
            res_list.append(self.kernels[i](x))
        return res_list
