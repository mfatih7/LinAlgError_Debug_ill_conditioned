import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

# Checkpointing can be used with a module or a function of a module

class Conv2d_N_REL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding ):
        super(Conv2d_N_REL, self).__init__()
        
        self.GNseperation = 8
        
        self.cnn = nn.Conv2d(    in_channels = in_channels,
                                 out_channels = out_channels,
                                 kernel_size = kernel_size,
                                 stride = stride,
                                 padding = padding,
                                 bias = False,)
                                 
        self.norm = nn.BatchNorm2d(out_channels)
            
        self.non_lin = nn.ReLU()
            
    def forward(self, x):        
        x = self.non_lin( self.norm( self.cnn(x) ) )
        return x

class Depth_Wise_Seperable_Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding ):
        super(Depth_Wise_Seperable_Convolution, self).__init__()
        
        self.GNseperation = 8
        
        depth_wise_conv_list = []
        
        depth_wise_conv_list.append( nn.Conv2d( in_channels = in_channels, out_channels = in_channels, kernel_size = kernel_size, stride = stride, padding = padding, groups = in_channels, bias = False, ) )
        
        self.depth_wise_conv = nn.Sequential(*depth_wise_conv_list)
        
        self.norm1 = nn.BatchNorm2d(in_channels)
                
        self.point_wise_conv = nn.Conv2d( in_channels = in_channels, out_channels = out_channels, kernel_size = (1,1), stride = (1,1), bias = False, )
        
        self.norm2 = nn.BatchNorm2d(out_channels)
                    
        self.non_lin = nn.ReLU()
            
    def forward(self, x):        
        
        x = self.non_lin( self.norm1( self.depth_wise_conv( x ) ) )
        x = self.non_lin( self.norm2( self.point_wise_conv( x ) ) )
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, size_reduction ):
        super(Block, self).__init__()
        
        layers = []
        
        if( size_reduction == 1 ):
        
            layers.append( Depth_Wise_Seperable_Convolution( in_channels=in_channels, out_channels=out_channels, kernel_size=(2,1), stride=(2,1), padding=(0,0) ) )
            
        elif( size_reduction == 0 ):
            layers.append( nn.ReflectionPad2d( (0, 0, 0, 1) ) ) # padding_left, padding_right, padding_top, padding_bottom
            layers.append( Depth_Wise_Seperable_Convolution( in_channels=in_channels, out_channels=out_channels, kernel_size=(2,1), stride=(1,1), padding=(0,0) ) )
            
        self.net = nn.Sequential(*layers)
            
    def forward(self, x):        
        x = self.net( x )        
        return x

class MobileNetV1(nn.Module):
    def __init__(self, N, en_grad_checkpointing ):
        super(MobileNetV1, self).__init__()
        
        self.N = N
        
        self.en_grad_checkpointing = en_grad_checkpointing
        
        self.n_blocks = int(math.log2(N))
        
        layers = []
        
        layers.append( Conv2d_N_REL( in_channels=2, out_channels=32, kernel_size=(1,4), stride=(1,1), padding=(0,0) ) )
        layers.append( Conv2d_N_REL( in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1), padding=(0,0) ) )
        
        layer_count_prev = layers[-1].cnn.out_channels
        
        for block_no in range(self.n_blocks):
            
            if((block_no+2)%2 < 1):
                in_channels = layer_count_prev
                out_channels = layer_count_prev
            else:
                in_channels = layer_count_prev
                out_channels = layer_count_prev*2
                layer_count_prev = layer_count_prev*2
            
            layers.append( Block( in_channels=in_channels, out_channels=out_channels, size_reduction=1 ) )
            if( block_no < self.n_blocks-1 ):
                layers.append( Block( in_channels=out_channels, out_channels=out_channels, size_reduction=0 ) )

        self.fully_connected_size = layers[-1].net[-1].point_wise_conv.out_channels
            
        self.fc1 = nn.Linear(self.fully_connected_size * 1 * 1, int(self.fully_connected_size/4) )
        
        self.fc2 = nn.Linear(int(self.fully_connected_size/4) * 1 * 1, 10)
        
        self.non_lin = nn.ReLU()  
        
        self.net = nn.Sequential(*layers)

    def block_1(self, x):
        
        x = self.net( x )
        
        return x        
        
    def forward(self, x):
        
        if(self.en_grad_checkpointing == False or self.training == False):
            
            x = self.block_1( x )
        else:
            
            x = checkpoint( self.block_1, x )
                
        x = x.view( -1, self.fully_connected_size * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x
        
def get_model( config, N, en_checkpointing ):   

    if( N == 512 or N == 1024 or N == 2048 or N == 4096 ):
    
        return MobileNetV1( N, en_checkpointing )
    
    else:
        print( 'Not Valid N ' + str(N) )
        assert(0)
