import torch
import torch.nn as nn
import math

class Conv2d_N(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, bias ):
        super(Conv2d_N, self).__init__()
        
        self.cnn = nn.Conv2d( in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, bias = bias, )
        
        self.norm = nn.BatchNorm2d(out_channels, track_running_stats=False)
            
    def forward(self, x):
        x = self.norm( self.cnn(x) )
            
        return x

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, height, non_lin ):
        super(Block, self).__init__()
        
        self.height_reduction = Conv2d_N( in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=(2,1),
                                          stride=(2,1),
                                          bias = False, )        
        height = int( height/2 )
        
        self.channel_mixing = Conv2d_N( in_channels=out_channels,
                                        out_channels=out_channels,
                                        kernel_size=(1,1),
                                        stride=(1,1),
                                        bias = False, )
        
        self.context_mixing = Conv2d_N( in_channels=height,
                                        out_channels=height,
                                        kernel_size=(1,1),
                                        stride=(1,1),
                                        bias = False, )
        
        if( non_lin == 'ReLU' ):
            self.non_lin = nn.ReLU()
        elif( non_lin == 'LeakyReLU' ):
            self.non_lin = nn.LeakyReLU()            
            
    def forward(self, x):
        
        x = self.height_reduction( x )
        x = self.non_lin( x )
        
        shortcut = x
        x = self.channel_mixing( x )
        x = self.non_lin( shortcut + x )
        
        x = x.permute(0, 2, 1, 3)
        
        shortcut = x
        x = self.context_mixing( x )
        x = self.non_lin( shortcut + x )
        
        x = x.permute(0, 2, 1, 3)

        return x

class CNN_Residual_Context(nn.Module):
    def __init__(self, N, model_width, non_lin ):
        super(CNN_Residual_Context, self).__init__()
        
        self.N = N
        self.model_width = model_width
        
        self.n_blocks = int(math.log2(N))
        
        height = self.N
        
        self.channel_reduction = Conv2d_N( in_channels=2, out_channels=32, kernel_size=(1, self.model_width), stride=(1,1), bias = False )
        layer_count_prev = 32
        
        layers = []
        
        for block_no in range(self.n_blocks):
            
            in_channels = layer_count_prev
            out_channels = int( layer_count_prev + ( layer_count_prev / 4 ) )
            
            layers.append( Block( in_channels=in_channels, out_channels=out_channels, height=height, non_lin=non_lin ) )
            
            height = int( height/2 )
            
            layer_count_prev = out_channels

        self.fully_connected_size = out_channels
            
        self.fc1 = nn.Linear(self.fully_connected_size * 1 * 1, int(self.fully_connected_size/4))
        
        self.fc2 = nn.Linear(int(self.fully_connected_size/4) * 1 * 1, 1)
        
        if( non_lin == 'ReLU' ):
            self.non_lin = nn.ReLU()
        elif( non_lin == 'LeakyReLU' ):
            self.non_lin = nn.LeakyReLU()         
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        
        x = self.non_lin( self.channel_reduction( x ) )
        
        x = self.net( x )
                
        x = x.view( -1, self.fully_connected_size * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x
        
def get_model( config, N, model_width, en_checkpointing ):   

    if( N == 512 or N == 1024 or N == 2048 or N == 4096 ):
        
        # non_lin = 'ReLU'
        non_lin = 'LeakyReLU'  
    
        return CNN_Residual_Context( N, model_width, non_lin )
    
    else:
        print( 'Not Valid N ' + str(N) )
        assert(0)
        
if __name__ == '__main__':
    
    import os    
    os.chdir( os.path.dirname( os.getcwd( ) ) )       
    from config import get_config
    from models.models import get_model_structure
    from models.models import get_model_params_and_FLOPS
    config = get_config()
    
    device = 'cpu'
    
    N = 512
    # N = 1024
    # N = 2048
    model_width = 4
    en_checkpointing = False
    
    first_model_no = 0
    last_model_no = 1    
    for i in range( first_model_no, last_model_no, 1 ):
        
        
        config.model_exp_no = i 
        print(f'config.model_exp_no: {config.model_exp_no}')
        model = get_model( config, N, model_width, en_checkpointing ).to(device)
        get_model_structure( config, device, model, N, model_width, en_grad_checkpointing=0)
        # get_model_params_and_FLOPS( config, device, model, N, model_width, en_grad_checkpointing=0)
        print('-'*80)
