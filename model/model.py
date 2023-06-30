import torch
import torch.nn as nn
import torch.nn.functional as F

class skipConnection(nn.Module):
 def __init__(self, nin, nout, padding): 
   super(skipConnection, self).__init__() 

   self.pointwise = nn.Sequential(
          nn.Conv2d(nin, nout, kernel_size=1, stride=2, padding=padding, dilation=2, bias=False),
          nn.BatchNorm2d(nout),
          nn.ReLU(),
          nn.BatchNorm2d(nout),
   )
  
 def forward(self, x): 
   out = self.pointwise(x) 
   return out
 



class depthwise_separable_conv(nn.Module):
 def __init__(self, nin, kernels_per_layer, nout): 
   super(depthwise_separable_conv, self).__init__() 
   
   self.depthwise = nn.Sequential(
          nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin, bias=False), 
          nn.BatchNorm2d(nin * kernels_per_layer),
          nn.ReLU()
   )

   self.pointwise = nn.Sequential(
          nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1, padding=1, bias=False),
          nn.BatchNorm2d(nout),
          nn.ReLU()
  )
  
 def forward(self, x): 
   out = self.depthwise(x) 
   out = self.pointwise(out) 
   return out
 



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        #self.dropout = nn.Dropout(dropout_value)

        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=9)
        ) 

        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            depthwise_separable_conv(3, 1, 32), 
            depthwise_separable_conv(32, 1, 32), 
            depthwise_separable_conv(32, 1, 32), 
        ) # output_size = 32

        # CONVOLUTION BLOCK 2
        self.convblock2 = nn.Sequential(
            depthwise_separable_conv(32, 1, 32), 
            depthwise_separable_conv(32, 1, 32), 
            depthwise_separable_conv(32, 1, 64),
        ) # output_size = 32

        # CONVOLUTION BLOCK 3
        self.convblock3 = nn.Sequential(
            depthwise_separable_conv(64, 1, 32), 
            depthwise_separable_conv(32, 1, 32),
            depthwise_separable_conv(32, 1, 64),
        ) # output_size = 10


        # CONVOLUTION BLOCK 4
        self.convblock4 = nn.Sequential(
            depthwise_separable_conv(64, 1, 64), 
            depthwise_separable_conv(64, 1, 128), 
            depthwise_separable_conv(128, 1, 256),  
        ) # output_size = 10



        # FINAL BLOCK
        self.finalBlock = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=100, kernel_size=(1, 1), padding=0, bias=False),
            nn.Conv2d(in_channels=100, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            
        ) # output_size =


        # TRANSITION BLOCK 1
        self.transblock1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=2, dilation=2, bias=False),
        ) # output_size = 16


        # TRANSITION BLOCK 2
        self.transblock2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=2, dilation=2, bias=False),
            nn.BatchNorm2d(64),
        ) # output_size = 8

        # TRANSITION BLOCK 3
        self.transblock3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=2, dilation=2, bias=False),
            nn.BatchNorm2d(64),
        ) # output_size = 8

        self.skipConnection1 = nn.Sequential(
          skipConnection(32, 64, 5)
        ) # output_size = 8

        self.skipConnection2 = nn.Sequential(
          skipConnection(64, 64, 5)
        ) # output_size = 8

        self.skipConnection3 = nn.Sequential(
          skipConnection(64, 256, 5)
        ) # output_size = 8


    def forward(self, x):
        x = self.convblock1(x) # 8x32x32
        x_skip = x.clone()
        x = self.transblock1(x) # 8x16x16

        x = self.convblock2(x) + self.skipConnection1(x_skip) # 16x32x32
        x_skip = x.clone()
        x = self.transblock2(x) # 8x16x16

        x = self.convblock3(x) + self.skipConnection2(x_skip)   # 16x16x16
        x_skip = x.clone()
        x = self.transblock3(x) # 8x16x16

        x = self.convblock4(x) + self.skipConnection3(x_skip)  # 16x16x16

        x = self.gap(x)
        x = self.finalBlock(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)