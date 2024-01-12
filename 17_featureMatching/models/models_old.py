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

class CNN_Basic_28_64(nn.Module):
    def __init__(self, en_grad_checkpointing):
        super(CNN_Basic_28_64, self).__init__()
        
        self.en_grad_checkpointing = en_grad_checkpointing
        
        #N, C=1, H=28*(2**6), W=28*(2**6)
#############################################################
        self.Conv2d_N_REL_A1 = Conv2d_N_REL( in_channels=1, out_channels=16, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        
        #N, C=16+8*0, H=28*(2**5), W=28*(2**5)
##############################################################
        self.Conv2d_N_REL_A2 = Conv2d_N_REL( in_channels=16, out_channels=24, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        
        #N, C=16+8*1, H=28*(2**4), W=28*(2**4)
##############################################################
        self.Conv2d_N_REL_A3 = Conv2d_N_REL( in_channels=24, out_channels=32, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        
        #N, C=16+8*2, H=28*(2**3), W=28*(2**3)
##############################################################
        self.Conv2d_N_REL_A4 = Conv2d_N_REL( in_channels=32, out_channels=40, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        
        #N, C=16+8*3, H=28*(2**2), W=28*(2**2)
##############################################################
        self.Conv2d_N_REL_A5 = Conv2d_N_REL( in_channels=40, out_channels=48, kernel_size=(2,2), stride=(2,2), padding=(0,0) )

        #N, C=16+8*4, H=28*(2**1), W=28*(2**1)
##############################################################
        self.Conv2d_N_REL_A6 = Conv2d_N_REL( in_channels=48, out_channels=56, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
      
        

        #N, C=16+8*5, H=28, W=28
##############################################################
        self.Conv2d_N_REL_1 = Conv2d_N_REL( in_channels=56, out_channels=64, kernel_size=(2,2), stride=(2,2), padding=(0,0) )

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
        
    def forward(self, x):
        
        if(self.en_grad_checkpointing == False or self.training == False):
            x = self.Conv2d_N_REL_A1(x)
            x = self.Conv2d_N_REL_A2(x)
            x = self.Conv2d_N_REL_A3(x)
            x = self.Conv2d_N_REL_A4(x)
            x = self.Conv2d_N_REL_A5(x)
            x = self.Conv2d_N_REL_A6(x)        

            x = self.Conv2d_N_REL_1(x)
            x = self.Conv2d_N_REL_2(x)
            x = self.Conv2d_N_REL_3(x)
            x = self.Conv2d_N_REL_4(x)
            x = self.Conv2d_N_REL_5(x)
        else:
            x = checkpoint( self.Conv2d_N_REL_A1, x )
            x = checkpoint( self.Conv2d_N_REL_A2, x )
            x = checkpoint( self.Conv2d_N_REL_A3, x )
            x = checkpoint( self.Conv2d_N_REL_A4, x )
            x = checkpoint( self.Conv2d_N_REL_A5, x )
            x = checkpoint( self.Conv2d_N_REL_A6, x )
            
            x = checkpoint( self.Conv2d_N_REL_1, x )
            x = checkpoint( self.Conv2d_N_REL_2, x )
            x = checkpoint( self.Conv2d_N_REL_3, x )
            x = checkpoint( self.Conv2d_N_REL_4, x )
            x = checkpoint( self.Conv2d_N_REL_5, x )
        
        x = x.view( -1, 96 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x
    
class CNN_Basic_28_32(nn.Module):
    def __init__(self, en_grad_checkpointing):
        super(CNN_Basic_28_32, self).__init__()
        
        self.en_grad_checkpointing = en_grad_checkpointing
        
        #N, C=1, H=28*(2**5), W=28*(2**5)
##############################################################
        self.Conv2d_N_REL_A2 = Conv2d_N_REL( in_channels=1, out_channels=24, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        
        #N, C=16+8*1, H=28*(2**4), W=28*(2**4)
##############################################################
        self.Conv2d_N_REL_A3 = Conv2d_N_REL( in_channels=24, out_channels=32, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        
        #N, C=16+8*2, H=28*(2**3), W=28*(2**3)
##############################################################
        self.Conv2d_N_REL_A4 = Conv2d_N_REL( in_channels=32, out_channels=40, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        
        #N, C=16+8*3, H=28*(2**2), W=28*(2**2)
##############################################################
        self.Conv2d_N_REL_A5 = Conv2d_N_REL( in_channels=40, out_channels=48, kernel_size=(2,2), stride=(2,2), padding=(0,0) )

        #N, C=16+8*4, H=28*(2**1), W=28*(2**1)
##############################################################
        self.Conv2d_N_REL_A6 = Conv2d_N_REL( in_channels=48, out_channels=56, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
      
        

        #N, C=16+8*5, H=28, W=28
##############################################################
        self.Conv2d_N_REL_1 = Conv2d_N_REL( in_channels=56, out_channels=64, kernel_size=(2,2), stride=(2,2), padding=(0,0) )

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
        
    def forward(self, x):
        
        if(self.en_grad_checkpointing == False or self.training == False):
            x = self.Conv2d_N_REL_A2(x)
            x = self.Conv2d_N_REL_A3(x)
            x = self.Conv2d_N_REL_A4(x)
            x = self.Conv2d_N_REL_A5(x)
            x = self.Conv2d_N_REL_A6(x)        

            x = self.Conv2d_N_REL_1(x)
            x = self.Conv2d_N_REL_2(x)
            x = self.Conv2d_N_REL_3(x)
            x = self.Conv2d_N_REL_4(x)
            x = self.Conv2d_N_REL_5(x)
        else:
            x = checkpoint( self.Conv2d_N_REL_A2, x )
            x = checkpoint( self.Conv2d_N_REL_A3, x )
            x = checkpoint( self.Conv2d_N_REL_A4, x )
            x = checkpoint( self.Conv2d_N_REL_A5, x )
            x = checkpoint( self.Conv2d_N_REL_A6, x )
            
            x = checkpoint( self.Conv2d_N_REL_1, x )
            x = checkpoint( self.Conv2d_N_REL_2, x )
            x = checkpoint( self.Conv2d_N_REL_3, x )
            x = checkpoint( self.Conv2d_N_REL_4, x )
            x = checkpoint( self.Conv2d_N_REL_5, x )
        
        x = x.view( -1, 96 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x
    
    
class CNN_Basic_28_16(nn.Module):
    def __init__(self, en_grad_checkpointing):
        super(CNN_Basic_28_16, self).__init__()
        
        self.en_grad_checkpointing = en_grad_checkpointing        
          
        #N, C=1, H=28*(2**4), W=28*(2**4)
##############################################################
        self.Conv2d_N_REL_A3 = Conv2d_N_REL( in_channels=1, out_channels=32, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        
        #N, C=16+8*2, H=28*(2**3), W=28*(2**3)
##############################################################
        self.Conv2d_N_REL_A4 = Conv2d_N_REL( in_channels=32, out_channels=40, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        
        #N, C=16+8*3, H=28*(2**2), W=28*(2**2)
##############################################################
        self.Conv2d_N_REL_A5 = Conv2d_N_REL( in_channels=40, out_channels=48, kernel_size=(2,2), stride=(2,2), padding=(0,0) )

        #N, C=16+8*4, H=28*(2**1), W=28*(2**1)
##############################################################
        self.Conv2d_N_REL_A6 = Conv2d_N_REL( in_channels=48, out_channels=56, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
      
        

        #N, C=16+8*5, H=28, W=28
##############################################################
        self.Conv2d_N_REL_1 = Conv2d_N_REL( in_channels=56, out_channels=64, kernel_size=(2,2), stride=(2,2), padding=(0,0) )

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
        
    def block(self, x):
        
        x = self.Conv2d_N_REL_A3( x )
        x = self.Conv2d_N_REL_A4( x )
        x = self.Conv2d_N_REL_A5( x )
        x = self.Conv2d_N_REL_A6( x )
        
        x = self.Conv2d_N_REL_1( x )
        x = self.Conv2d_N_REL_2( x )
        x = self.Conv2d_N_REL_3( x )
        x = self.Conv2d_N_REL_4( x )
        x = self.Conv2d_N_REL_5( x )
        
        # x = x.view( -1, 96 * 1 * 1 )
        # x = self.non_lin( self.fc1(x) )
        # x = self.fc2(x)
        
        return x
    
    def block_1(self, x):
        
        x = self.Conv2d_N_REL_A3( x )
        x = self.Conv2d_N_REL_A4( x )
        x = self.Conv2d_N_REL_A5( x )
        x = self.Conv2d_N_REL_A6( x )     
        
        return x
    
    def block_2(self, x):
        
        x = self.Conv2d_N_REL_1( x )
        x = self.Conv2d_N_REL_2( x )
        x = self.Conv2d_N_REL_3( x )
        return x
    
    def block_3(self, x):        
        
        x = self.Conv2d_N_REL_4( x )
        x = self.Conv2d_N_REL_5( x )
        return x
        
    def forward(self, x):

        if(self.en_grad_checkpointing == False or self.training == False):
            x = self.Conv2d_N_REL_A3(x)
            x = self.Conv2d_N_REL_A4(x)
            x = self.Conv2d_N_REL_A5(x)
            x = self.Conv2d_N_REL_A6(x)

            x = self.Conv2d_N_REL_1(x)
            x = self.Conv2d_N_REL_2(x)
            x = self.Conv2d_N_REL_3(x)
            x = self.Conv2d_N_REL_4(x)
            x = self.Conv2d_N_REL_5(x)

            # x = x.view( -1, 96 * 1 * 1 )
            # x = self.non_lin( self.fc1(x) )
            # x = self.fc2(x)

        else:            
            # x = checkpoint( self.block , x )
            
            x = checkpoint( self.block_1 , x )
            x = checkpoint( self.block_2 , x )
            x = checkpoint( self.block_3 , x )
        
        x = x.view( -1, 96 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x


class CNN_Basic_28_8(nn.Module):
    def __init__(self, en_grad_checkpointing):
        super(CNN_Basic_28_8, self).__init__()
        
        self.en_grad_checkpointing = en_grad_checkpointing
        
        #N, C=1, H=28*(2**3), W=28*(2**3)
##############################################################
        self.Conv2d_N_REL_A4 = Conv2d_N_REL( in_channels=1, out_channels=40, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
        
        #N, C=16+8*3, H=28*(2**2), W=28*(2**2)
##############################################################
        self.Conv2d_N_REL_A5 = Conv2d_N_REL( in_channels=40, out_channels=48, kernel_size=(2,2), stride=(2,2), padding=(0,0) )

        #N, C=16+8*4, H=28*(2**1), W=28*(2**1)
##############################################################
        self.Conv2d_N_REL_A6 = Conv2d_N_REL( in_channels=48, out_channels=56, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
      
        

        #N, C=16+8*5, H=28, W=28
##############################################################
        self.Conv2d_N_REL_1 = Conv2d_N_REL( in_channels=56, out_channels=64, kernel_size=(2,2), stride=(2,2), padding=(0,0) )

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
        
    def forward(self, x):

        if(self.en_grad_checkpointing == False or self.training == False):
            x = self.Conv2d_N_REL_A4(x)
            x = self.Conv2d_N_REL_A5(x)
            x = self.Conv2d_N_REL_A6(x)        

            x = self.Conv2d_N_REL_1(x)
            x = self.Conv2d_N_REL_2(x)
            x = self.Conv2d_N_REL_3(x)
            x = self.Conv2d_N_REL_4(x)
            x = self.Conv2d_N_REL_5(x)
        else:
            x = checkpoint( self.Conv2d_N_REL_A4, x )
            x = checkpoint( self.Conv2d_N_REL_A5, x )
            x = checkpoint( self.Conv2d_N_REL_A6, x )
            
            x = checkpoint( self.Conv2d_N_REL_1, x )
            x = checkpoint( self.Conv2d_N_REL_2, x )
            x = checkpoint( self.Conv2d_N_REL_3, x )
            x = checkpoint( self.Conv2d_N_REL_4, x )
            x = checkpoint( self.Conv2d_N_REL_5, x )  
        
        x = x.view( -1, 96 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x


class CNN_Basic_28_4(nn.Module):
    def __init__(self, en_grad_checkpointing):
        super(CNN_Basic_28_4, self).__init__()
        
        self.en_grad_checkpointing = en_grad_checkpointing
        
        #N, C=1, H=28*(2**2), W=28*(2**2)
##############################################################
        self.Conv2d_N_REL_A5 = Conv2d_N_REL( in_channels=1, out_channels=48, kernel_size=(2,2), stride=(2,2), padding=(0,0) )

        #N, C=16+8*4, H=28*(2**1), W=28*(2**1)
##############################################################
        self.Conv2d_N_REL_A6 = Conv2d_N_REL( in_channels=48, out_channels=56, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
      
        

        #N, C=16+8*5, H=28, W=28
##############################################################
        self.Conv2d_N_REL_1 = Conv2d_N_REL( in_channels=56, out_channels=64, kernel_size=(2,2), stride=(2,2), padding=(0,0) )

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
        
    def forward(self, x):

        if(self.en_grad_checkpointing == False or self.training == False):
            x = self.Conv2d_N_REL_A5(x)
            x = self.Conv2d_N_REL_A6(x)        

            x = self.Conv2d_N_REL_1(x)
            x = self.Conv2d_N_REL_2(x)
            x = self.Conv2d_N_REL_3(x)
            x = self.Conv2d_N_REL_4(x)
            x = self.Conv2d_N_REL_5(x)
        else:
            x = checkpoint( self.Conv2d_N_REL_A5, x )
            x = checkpoint( self.Conv2d_N_REL_A6, x )
            
            x = checkpoint( self.Conv2d_N_REL_1, x )
            x = checkpoint( self.Conv2d_N_REL_2, x )
            x = checkpoint( self.Conv2d_N_REL_3, x )
            x = checkpoint( self.Conv2d_N_REL_4, x )
            x = checkpoint( self.Conv2d_N_REL_5, x )   
        
        x = x.view( -1, 96 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x

class CNN_Basic_28_2(nn.Module):
    def __init__(self, en_grad_checkpointing):
        super(CNN_Basic_28_2, self).__init__()
        
        self.en_grad_checkpointing = en_grad_checkpointing
        
        #N, C=1, H=28*(2**1), W=28*(2**1)
##############################################################
        self.Conv2d_N_REL_A6 = Conv2d_N_REL( in_channels=1, out_channels=56, kernel_size=(2,2), stride=(2,2), padding=(0,0) )
      
        

        #N, C=16+8*5, H=28, W=28
##############################################################
        self.Conv2d_N_REL_1 = Conv2d_N_REL( in_channels=56, out_channels=64, kernel_size=(2,2), stride=(2,2), padding=(0,0) )

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
        
    def block(self, x, conv):
        x = conv(x)
        return x
        
    def forward(self, x):
        
        if(self.en_grad_checkpointing == False or self.training == False):
            x = self.Conv2d_N_REL_A6(x)        

            x = self.Conv2d_N_REL_1(x)
            x = self.Conv2d_N_REL_2(x)
            x = self.Conv2d_N_REL_3(x)
            x = self.Conv2d_N_REL_4(x)
            x = self.Conv2d_N_REL_5(x)
        else:            
            x = checkpoint( self.Conv2d_N_REL_A6, x )
            
            x = checkpoint( self.Conv2d_N_REL_1, x )
            x = checkpoint( self.Conv2d_N_REL_2, x )
            x = checkpoint( self.Conv2d_N_REL_3, x )
            x = checkpoint( self.Conv2d_N_REL_4, x )
            x = checkpoint( self.Conv2d_N_REL_5, x )  
        
        x = x.view( -1, 96 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x    
    
class CNN_Basic_28_1(nn.Module):
    def __init__(self, en_grad_checkpointing):
        super(CNN_Basic_28_1, self).__init__()
        
        self.en_grad_checkpointing = en_grad_checkpointing

        #N, C=1, H=28, W=28
##############################################################
        self.Conv2d_N_REL_1 = Conv2d_N_REL( in_channels=1, out_channels=64, kernel_size=(2,2), stride=(2,2), padding=(0,0) )

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
        
    def forward(self, x):
        
        if(self.en_grad_checkpointing == False or self.training == False):
            x = self.Conv2d_N_REL_1(x)
            x = self.Conv2d_N_REL_2(x)
            x = self.Conv2d_N_REL_3(x)
            x = self.Conv2d_N_REL_4(x)
            x = self.Conv2d_N_REL_5(x)
        else:            
            x = checkpoint( self.block, x, self.Conv2d_N_REL_1 )
            x = checkpoint( self.block, x, self.Conv2d_N_REL_2 )
            x = checkpoint( self.block, x, self.Conv2d_N_REL_3 )
            x = checkpoint( self.block, x, self.Conv2d_N_REL_4 )
            x = checkpoint( self.block, x, self.Conv2d_N_REL_5 )

        x = x.view( -1, 96 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x

class CNN_Basic(nn.Module):
    def __init__(self):
        super(CNN_Basic, self).__init__()

        #N, C=1, H=28, W=28
##############################################################

        self.cnn1 = nn.Conv2d( in_channels=1, out_channels=16, kernel_size=(3,3), stride=(1,1), padding=(1,1) )           
        self.batchnorm1 = nn.BatchNorm2d(16)

        #N, C=16, H=28, W=28        
##############################################################

        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnorm2 = nn.BatchNorm2d(24)        
        
        #N, C=32, H=14, W=14        
##############################################################

        self.cnn3 = nn.Conv2d( in_channels=24, out_channels=32, kernel_size=(3,3), stride=(1,1), padding=(1,1) )           
        self.batchnorm3 = nn.BatchNorm2d(32)

        #N, C=64, H=14, W=14        
##############################################################

        self.cnn4 = nn.Conv2d(in_channels=32, out_channels=40, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnorm4 = nn.BatchNorm2d(40)        
        
        #N, C=128, H=7, W=7        
##############################################################

        self.cnn5 = nn.Conv2d( in_channels=40, out_channels=48, kernel_size=(3,3), stride=(1,1), padding=(1,1) )           
        self.batchnorm5 = nn.BatchNorm2d(48)

        #N, C=256, H=7, W=7        
##############################################################

        self.cnn6 = nn.Conv2d(in_channels=48, out_channels=56, kernel_size=(2,2), stride=(2,2), padding=(1,1) )           
        self.batchnorm6 = nn.BatchNorm2d(56)        
        
        #N, C=512, H=4, W=4        
##############################################################

        self.cnn7 = nn.Conv2d(in_channels=56, out_channels=64, kernel_size=(2,2), stride=(2,2), padding=(0,0) )           
        self.batchnorm7 = nn.BatchNorm2d(64)        
        
        #N, C=1024, H=2, W=2        
##############################################################

        self.cnn8 = nn.Conv2d(in_channels=64, out_channels=72, kernel_size=(2,2), stride=(1,1), padding=(0,0) )           
        self.batchnorm8 = nn.BatchNorm2d(72)        
        
        #N, C=1024, H=1, W=1        
##############################################################  

        self.fc1 = nn.Linear(72 * 1 * 1, 36)   
        
        #N, C=128, H=1, W=1
##############################################################    

        self.fc2 = nn.Linear(36 * 1 * 1, 10)
        
        self.non_lin = nn.ReLU()
        
    def forward(self, x):
        
        # x = x.half()
        
        x = self.non_lin( self.batchnorm1( self.cnn1(x) ) )
        x = self.non_lin( self.batchnorm2( self.cnn2(x) ) )
        x = self.non_lin( self.batchnorm3( self.cnn3(x) ) )
        x = self.non_lin( self.batchnorm4( self.cnn4(x) ) )
        x = self.non_lin( self.batchnorm5( self.cnn5(x) ) )
        
        # x = x.float()
        
        x = self.non_lin( self.batchnorm6( self.cnn6(x) ) )
        x = self.non_lin( self.batchnorm7( self.cnn7(x) ) )
        x = self.non_lin( self.batchnorm8( self.cnn8(x) ) )
        
        x = x.view( -1, 72 * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x

def get_model( config, input_expand_ratio, en_checkpointing ):
    
    if( input_expand_ratio == 64):    
        return CNN_Basic_28_64( en_checkpointing )
    elif( input_expand_ratio == 32):    
        return CNN_Basic_28_32( en_checkpointing )
    elif( input_expand_ratio == 16):    
        return CNN_Basic_28_16( en_checkpointing )
    elif( input_expand_ratio == 8):    
        return CNN_Basic_28_8( en_checkpointing )
    elif( input_expand_ratio == 4):    
        return CNN_Basic_28_4( en_checkpointing )
    elif( input_expand_ratio == 2):    
        return CNN_Basic_28_2( en_checkpointing )
    elif( input_expand_ratio == 1):    
        return CNN_Basic_28_1( en_checkpointing )
    # elif( input_expand_ratio == 1):    
    #     return CNN_Basic( )
    else:
        print( 'Not Valid Input Expand Ratio ' + str(input_expand_ratio) )
        assert(0)


