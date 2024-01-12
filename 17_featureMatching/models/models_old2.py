import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Checkpointing can be used with a module or a function of a module

class Conv2d_N_REL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2d_N_REL, self).__init__()
        
        self.GNseperation = 8
        
        self.cnn = nn.Conv2d(    in_channels = in_channels,
                                 out_channels = out_channels,
                                 kernel_size = kernel_size,
                                 stride = stride,
                                 padding = padding, )
                                 
        self.norm = nn.BatchNorm2d(out_channels)
            
        self.non_lin = nn.ReLU()
            
    def forward(self, x):
        
        x = self.non_lin( self.norm( self.cnn(x) ) )        
        return x
    
class CNN_Basic_28(nn.Module):
    def __init__(self, en_grad_checkpointing, input_expand_ratio):
        super(CNN_Basic_28, self).__init__()
        
        self.en_grad_checkpointing = en_grad_checkpointing
        self.input_expand_ratio = input_expand_ratio

        #N, C=8*0+1, H=28*(2**6), W=28*(2**6)
#############################################################
        self.Conv2d_N_REL_A1_1 = Conv2d_N_REL( in_channels=1, out_channels=8, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        
        #N, C=8*1, H=28*(2**6), W=28*(2**6)
#############################################################
        self.Conv2d_N_REL_A2 = Conv2d_N_REL( in_channels=8, out_channels=16, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        self.Conv2d_N_REL_A2_1 = Conv2d_N_REL( in_channels=1, out_channels=16, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        
        #N, C=8*2, H=28*(2**5), W=28*(2**5)
##############################################################
        self.Conv2d_N_REL_A3 = Conv2d_N_REL( in_channels=16, out_channels=24, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        self.Conv2d_N_REL_A3_1 = Conv2d_N_REL( in_channels=1, out_channels=24, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        
        #N, C=8*3, H=28*(2**4), W=28*(2**4)
##############################################################
        self.Conv2d_N_REL_A4 = Conv2d_N_REL( in_channels=24, out_channels=32, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        self.Conv2d_N_REL_A4_1 = Conv2d_N_REL( in_channels=1, out_channels=32, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        
        #N, C=8*4, H=28*(2**3), W=28*(2**3)
##############################################################
        self.Conv2d_N_REL_A5 = Conv2d_N_REL( in_channels=32, out_channels=40, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        self.Conv2d_N_REL_A5_1 = Conv2d_N_REL( in_channels=1, out_channels=40, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        
        #N, C=8*5, H=28*(2**2), W=28*(2**2)
##############################################################
        self.Conv2d_N_REL_A6 = Conv2d_N_REL( in_channels=40, out_channels=48, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        self.Conv2d_N_REL_A6_1 = Conv2d_N_REL( in_channels=1, out_channels=48, kernel_size=(2,2), stride=(2,2), padding=(0,0) )

        #N, C=8*6, H=28*(2**1), W=28*(2**1)
##############################################################
        self.Conv2d_N_REL_A7 = Conv2d_N_REL( in_channels=48, out_channels=56, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        self.Conv2d_N_REL_A7_1 = Conv2d_N_REL( in_channels=1, out_channels=56, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
      
        

        #N, C=16+8*5, H=28, W=28
##############################################################
        self.Conv2d_N_REL_1 = Conv2d_N_REL( in_channels=56, out_channels=64, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        self.Conv2d_N_REL_1_1 = Conv2d_N_REL( in_channels=1, out_channels=64, kernel_size=(2,2), stride=(2,2), padding=(0,0) )

        #N, C=16+8*5, H=14, W=14        
##############################################################
        self.Conv2d_N_REL_2 = Conv2d_N_REL( in_channels=64, out_channels=72, kernel_size=(2,2), stride=(2,2), padding=(0,0) )  
        
        #N, C=16+8*6, H=7, W=7        
##############################################################
        self.Conv2d_N_REL_3 = Conv2d_N_REL( in_channels=72, out_channels=80, kernel_size=(2,2), stride=(2,2), padding=(1,1) )

        #N, C=16+8*7, H=4, W=4  
##############################################################
        self.Conv2d_N_REL_4 = Conv2d_N_REL( in_channels=80, out_channels=88, kernel_size=(2,2), stride=(2,2), padding=(0,0) )      
        
        #N, C=16+8*8, H=2, W=2  
##############################################################
        self.Conv2d_N_REL_5 = Conv2d_N_REL( in_channels=88, out_channels=96, kernel_size=(2,2), stride=(2,2), padding=(0,0) )


        #N, C=16+8*9, H=1, W=1        
############################################################## 
        self.fc1 = nn.Linear(96 * 1 * 1, 32)   
        
        #N, C=32, H=1, W=1
##############################################################
        self.fc2 = nn.Linear(32 * 1 * 1, 10)
        
        self.non_lin = nn.ReLU()
        
    def block_1(self, x):
        
        x = self.Conv2d_N_REL_1_1(x)
        x = self.Conv2d_N_REL_2(x)
        x = self.Conv2d_N_REL_3(x)
        x = self.Conv2d_N_REL_4(x)
        x = self.Conv2d_N_REL_5(x)
        
        x = x.view( -1, 96 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x

    def block_2(self, x):
        
        x = self.Conv2d_N_REL_A7_1(x)
        
        x = self.Conv2d_N_REL_1(x)
        x = self.Conv2d_N_REL_2(x)
        x = self.Conv2d_N_REL_3(x)
        x = self.Conv2d_N_REL_4(x)
        x = self.Conv2d_N_REL_5(x)
        
        x = x.view( -1, 96 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x

    def block_4(self, x):
        
        x = self.Conv2d_N_REL_A6_1(x)
        x = self.Conv2d_N_REL_A7(x)
        
        x = self.Conv2d_N_REL_1(x)
        x = self.Conv2d_N_REL_2(x)
        x = self.Conv2d_N_REL_3(x)
        x = self.Conv2d_N_REL_4(x)
        x = self.Conv2d_N_REL_5(x)
        
        x = x.view( -1, 96 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x
    
    def block_8(self, x):
        
        x = self.Conv2d_N_REL_A5_1(x)
        x = self.Conv2d_N_REL_A6(x)
        x = self.Conv2d_N_REL_A7(x)
        
        x = self.Conv2d_N_REL_1(x)
        x = self.Conv2d_N_REL_2(x)
        x = self.Conv2d_N_REL_3(x)
        x = self.Conv2d_N_REL_4(x)
        x = self.Conv2d_N_REL_5(x)
        
        x = x.view( -1, 96 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x

    def block_16(self, x):
        
        x = self.Conv2d_N_REL_A4_1(x)
        x = self.Conv2d_N_REL_A5(x)
        x = self.Conv2d_N_REL_A6(x)
        x = self.Conv2d_N_REL_A7(x)
        
        x = self.Conv2d_N_REL_1(x)
        x = self.Conv2d_N_REL_2(x)
        x = self.Conv2d_N_REL_3(x)
        x = self.Conv2d_N_REL_4(x)
        x = self.Conv2d_N_REL_5(x)
        
        x = x.view( -1, 96 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x

    def block_32(self, x):
        
        x = self.Conv2d_N_REL_A3_1(x)
        x = self.Conv2d_N_REL_A4(x)
        x = self.Conv2d_N_REL_A5(x)
        x = self.Conv2d_N_REL_A6(x)
        x = self.Conv2d_N_REL_A7(x)
        
        x = self.Conv2d_N_REL_1(x)
        x = self.Conv2d_N_REL_2(x)
        x = self.Conv2d_N_REL_3(x)
        x = self.Conv2d_N_REL_4(x)
        x = self.Conv2d_N_REL_5(x)
        
        x = x.view( -1, 96 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x
    
    def block_64(self, x):
        
        x = self.Conv2d_N_REL_A2_1(x)
        x = self.Conv2d_N_REL_A3(x)
        x = self.Conv2d_N_REL_A4(x)
        x = self.Conv2d_N_REL_A5(x)
        x = self.Conv2d_N_REL_A6(x)
        x = self.Conv2d_N_REL_A7(x)
        
        x = self.Conv2d_N_REL_1(x)
        x = self.Conv2d_N_REL_2(x)
        x = self.Conv2d_N_REL_3(x)
        x = self.Conv2d_N_REL_4(x)
        x = self.Conv2d_N_REL_5(x)
        
        x = x.view( -1, 96 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x
    
    def block_128(self, x):
        
        x = self.Conv2d_N_REL_A1_1(x)
        x = self.Conv2d_N_REL_A2(x)
        x = self.Conv2d_N_REL_A3(x)
        x = self.Conv2d_N_REL_A4(x)
        x = self.Conv2d_N_REL_A5(x)
        x = self.Conv2d_N_REL_A6(x)
        x = self.Conv2d_N_REL_A7(x)
        
        x = self.Conv2d_N_REL_1(x)
        x = self.Conv2d_N_REL_2(x)
        x = self.Conv2d_N_REL_3(x)
        x = self.Conv2d_N_REL_4(x)
        x = self.Conv2d_N_REL_5(x)
        
        x = x.view( -1, 96 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x
            
    def forward(self, x):
        
        if(self.en_grad_checkpointing == False or self.training == False):
            
            if( self.input_expand_ratio == 1 ):
                x = self.block_1( x )
            elif( self.input_expand_ratio == 2 ):
                x = self.block_2( x )
            elif( self.input_expand_ratio == 4 ):
                x = self.block_4( x )
            elif( self.input_expand_ratio == 8 ):
                x = self.block_8( x )
            elif( self.input_expand_ratio == 16 ):
                x = self.block_16( x )
            elif( self.input_expand_ratio == 32 ):
                x = self.block_32( x )
            elif( self.input_expand_ratio == 64 ):
                x = self.block_64( x )
            elif( self.input_expand_ratio == 128 ):
                x = self.block_128( x )
        else:
            if( self.input_expand_ratio == 1 ):
                x = checkpoint( self.block_1, x )
            elif( self.input_expand_ratio == 2 ):
                x = checkpoint( self.block_2, x )
            elif( self.input_expand_ratio == 4 ):
                x = checkpoint( self.block_4, x )
            elif( self.input_expand_ratio == 8 ):
                x = checkpoint( self.block_8, x )
            elif( self.input_expand_ratio == 16 ):
                x = checkpoint( self.block_16, x )
            elif( self.input_expand_ratio == 32 ):
                x = checkpoint( self.block_32, x )
            elif( self.input_expand_ratio == 64 ):
                x = checkpoint( self.block_64, x )
            elif( self.input_expand_ratio == 128 ):
                x = checkpoint( self.block_128, x )
        
        return x


def get_model( config, input_expand_ratio, en_checkpointing ):   

    if( input_expand_ratio == 1 or input_expand_ratio==2 or input_expand_ratio == 4 or input_expand_ratio == 8 or
        input_expand_ratio == 16 or input_expand_ratio ==32 or input_expand_ratio == 64 or input_expand_ratio == 128 ):
    
        return CNN_Basic_28( en_checkpointing, input_expand_ratio )
    
    else:
        print( 'Not Valid Input Expand Ratio ' + str(input_expand_ratio) )
        assert(0)
