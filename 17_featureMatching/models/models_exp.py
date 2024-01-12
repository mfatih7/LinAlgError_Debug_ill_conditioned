import torch
import torch.nn as nn
import math

class Context_Norm_1_to_1(nn.Module):   # Makes training 10% slower
    def __init__(self, eps=1e-5):
        super(Context_Norm_1_to_1, self).__init__()
        self.eps = eps

    def forward(self, activation_map):
        mean = activation_map.mean(dim=0, keepdim=True)
        std = activation_map.std(dim=0, keepdim=True) + self.eps

        normalized_map = (activation_map - mean) / std
        return normalized_map

class Conv2d_N(nn.Module):
    def __init__(self, in_channels, out_channels, height, kernel_size, stride, bias, enable_context_norm ):
        super(Conv2d_N, self).__init__()
        
        self.enable_context_norm = enable_context_norm
        
        self.cnn = nn.Conv2d( in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, bias = bias, )
        
        self.norm = nn.BatchNorm2d(out_channels, track_running_stats=False)
            
        if(self.enable_context_norm):
            if( height/int(kernel_size[0]) > 1 and kernel_size[1]==1 ):
                # self.cont_norm = nn.InstanceNorm2d(out_channels, eps=1e-3)  # does not hel for 1 to 1 training
                self.cont_norm = Context_Norm_1_to_1()
            else:
                self.cont_norm = nn.Identity()
            
    def forward(self, x):
        if(self.enable_context_norm):
            x = self.norm( self.cont_norm( self.cnn(x) ) )
        else:
            x = self.norm( self.cnn(x) )
        return x
    
class Width_Reduction(nn.Module):
    def __init__(self, in_width, out_channels, height, enable_context_norm, non_lin):
        super(Width_Reduction, self).__init__()
        
        self.width_reduction = Conv2d_N( in_channels = 2, out_channels = out_channels, height = height, kernel_size = (1, in_width), stride = (1,1), bias = False, enable_context_norm = enable_context_norm )
        
        if( non_lin == 'ReLU' ):
            self.non_lin = nn.ReLU()
        elif( non_lin == 'LeakyReLU' ):
            self.non_lin = nn.LeakyReLU()
            
    def forward(self, x):
        x = self.non_lin( self.width_reduction(x) )
        return x
    
class Height_Reduction(nn.Module):
    def __init__(self, in_channels, out_channels, height, enable_context_norm, non_lin):
        super(Height_Reduction, self).__init__()
        
        self.height_reduction = Conv2d_N( in_channels = in_channels, out_channels = out_channels, height = height, kernel_size = (2,1), stride = (2,1), bias = False, enable_context_norm = enable_context_norm )
        
        if( non_lin == 'ReLU' ):
            self.non_lin = nn.ReLU()
        elif( non_lin == 'LeakyReLU' ):
            self.non_lin = nn.LeakyReLU()
            
    def forward(self, x):
        x = self.non_lin( self.height_reduction(x) )
        return x
    
class Channel_Reduction(nn.Module):
    def __init__(self, in_channels, out_channels, height, enable_context_norm, non_lin):
        super(Channel_Reduction, self).__init__()
        
        self.channel_reduction = Conv2d_N( in_channels = in_channels, out_channels = out_channels, height = height, kernel_size = (1,1), stride = (1,1), bias = False, enable_context_norm = enable_context_norm )
        
        if( non_lin == 'ReLU' ):
            self.non_lin = nn.ReLU()
        elif( non_lin == 'LeakyReLU' ):
            self.non_lin = nn.LeakyReLU()
            
    def forward(self, x):
        x = self.non_lin( self.channel_reduction(x) )
        return x
    
class Pointwise_Conv_Shortcut(nn.Module):
    def __init__(self, channels, height, enable_context_norm, non_lin):
        super(Pointwise_Conv_Shortcut, self).__init__()
        
        self.pointwise_conv = Conv2d_N( in_channels = channels, out_channels = channels, height = height, kernel_size = (1,1), stride = (1,1), bias = False, enable_context_norm = enable_context_norm )
        
        if( non_lin == 'ReLU' ):
            self.non_lin = nn.ReLU()
        elif( non_lin == 'LeakyReLU' ):
            self.non_lin = nn.LeakyReLU()
            
    def forward(self, x):
        shortcut = x
        x = self.pointwise_conv(x)
        x = self.non_lin( shortcut + x )
        return x
    
class Depth_Wise_Seperable_Convolution(nn.Module):
    def __init__(self, in_channels, out_channels, enable_context_norm, non_lin):
        super(Depth_Wise_Seperable_Convolution, self).__init__()
        
        self.depth_wise_conv = nn.Conv2d( in_channels = in_channels, out_channels = in_channels, kernel_size = (2,1), stride = (2,1), groups = in_channels, bias = False )
        
        ## enable_context_norm can be added
                
        self.norm_1 = nn.BatchNorm2d( in_channels, track_running_stats=False )
        
        self.point_wise_conv = nn.Conv2d( in_channels = in_channels, out_channels = out_channels, kernel_size = (1,1), stride = (1,1), bias = False )
        
        ## enable_context_norm can be added
        
        self.norm_2 = nn.BatchNorm2d( out_channels, track_running_stats=False )
            
        if( non_lin == 'ReLU' ):
            self.non_lin = nn.ReLU()
        elif( non_lin == 'LeakyReLU' ):
            self.non_lin = nn.LeakyReLU()
            
    def forward(self, x):        
        
        x = self.non_lin( self.norm_1( self.depth_wise_conv( x ) ) )
        x = self.non_lin( self.norm_2( self.point_wise_conv( x ) ) )
        return x

class Block(nn.Module):
    def __init__(self, block_no, in_width, channels_0, channels_1, channels_2, height, channel_reduction_layer_enable, enable_context_norm, non_lin, pointwise_conv_count, height_reduction_type):
        super(Block, self).__init__()
                
        self.block_no = block_no
        
        self.pointwise_conv_count = pointwise_conv_count
        
        self.channel_reduction_layer_enable = channel_reduction_layer_enable            
        
        if(block_no==0):            
            self.width_reduction = Width_Reduction(   in_width = in_width,
                                                      out_channels = channels_1,
                                                      height = height,
                                                      enable_context_norm = enable_context_norm,
                                                      non_lin = non_lin )
        else:
            
            if(height_reduction_type=='normal_conv'):
            
                self.height_reduction = Height_Reduction( in_channels = channels_0,
                                                          out_channels = channels_1,
                                                          height = height,
                                                          enable_context_norm = enable_context_norm,
                                                          non_lin = non_lin )
            elif(height_reduction_type=='depth_wise_sep_conv'):
            
                self.height_reduction = Depth_Wise_Seperable_Convolution( in_channels = channels_0,
                                                                          out_channels = channels_1,
                                                                          enable_context_norm = enable_context_norm,
                                                                          non_lin = non_lin )
            height = int( height/2 )
        
        if(channel_reduction_layer_enable):
            self.channel_reduction = Channel_Reduction( in_channels = channels_1, 
                                                        out_channels = channels_2,
                                                        height = height,
                                                        enable_context_norm = enable_context_norm,
                                                        non_lin = non_lin, )

        if( pointwise_conv_count > 0 ):
            pointwise_conv_layers = []
            
            for lay in range( pointwise_conv_count ):
                pointwise_conv_layers.append( Pointwise_Conv_Shortcut( channels = channels_2,
                                                                       height = height,
                                                                       enable_context_norm = enable_context_norm,
                                                                       non_lin = non_lin ) )
            self.net = nn.Sequential(*pointwise_conv_layers)
        
    def forward(self, x):
        
        if( self.block_no == 0 ):
            x = self.width_reduction( x )
        else:
            x = self.height_reduction( x )
            
        if( self.channel_reduction_layer_enable ):
            x = self.channel_reduction( x )
        
        if( self.pointwise_conv_count > 0 ):
            x = self.net( x )
            
        return x
    
class model_exp_00(nn.Module):
    def __init__(self, N,
                       in_width,
                       init_channel_count,
                       ch_expans_param,
                       height_reduction_type,
                       channel_reduction_layer_enable,
                       channel_reduction_ratio,
                       pointwise_conv_layers_count,
                       pointwise_conv_layers_param,
                       enable_context_norm,
                       non_lin, ):
        super(model_exp_00, self).__init__()        
        
        self.in_width = in_width
        self.init_channel_count = init_channel_count
        self.ch_expans_param = ch_expans_param
        self.channel_reduction_ratio = channel_reduction_ratio
        self.pointwise_conv_layers_count = pointwise_conv_layers_count
        self.pointwise_conv_layers_param = pointwise_conv_layers_param
        
        self.n_blocks = int( math.log2(N) + 1 )

        height = N
        
        block_in_channel_counts, block_out_channel_counts = self.calculate_block_channel_counts()        
        
        layers = []
        
        for block_no in range(self.n_blocks):
            
            if( pointwise_conv_layers_count > 0 ):
                pointwise_conv_count = 1
            else:
                pointwise_conv_count = 0
            
            if(block_no==0):
                channels_0 = 2
            else:
                channels_0 = block_out_channel_counts[block_no-1]
            
            layers.append( Block( block_no = block_no,
                                  in_width = self.in_width,
                                  channels_0 = channels_0,
                                  channels_1 = block_in_channel_counts[block_no],
                                  channels_2 = block_out_channel_counts[block_no],
                                  height = height,
                                  channel_reduction_layer_enable = channel_reduction_layer_enable,
                                  enable_context_norm = enable_context_norm,
                                  non_lin = non_lin,
                                  pointwise_conv_count = pointwise_conv_count,
                                  height_reduction_type = height_reduction_type, ) )
            if(block_no>0):
                height = int( height / 2 )            
            
        self.net = nn.Sequential(*layers)
        
        self.initial_fully_connected_size = block_out_channel_counts[-1]
        
        self.fc1 = nn.Linear(self.initial_fully_connected_size * 1 * 1, int(self.initial_fully_connected_size/4))
        
        self.fc2 = nn.Linear( int(self.initial_fully_connected_size/4) * 1 * 1, 1)
        
        if( non_lin == 'ReLU' ):
            self.non_lin = nn.ReLU()
        elif( non_lin == 'LeakyReLU' ):
            self.non_lin = nn.LeakyReLU()
            
    def calculate_block_channel_counts(self):
        
        n0_matches = torch.zeros( (self.n_blocks), dtype=torch.int32 )
        n1_info_size = torch.zeros( (self.n_blocks), dtype=torch.int32 )
        n2_info_size_with_channels_raw = torch.zeros( (self.n_blocks), dtype=torch.int32 )
        n3_info_size_with_channels_processed = torch.zeros( (self.n_blocks), dtype=torch.float32 )
        n4_info_size_with_channels_processed_reduced = torch.zeros( (self.n_blocks), dtype=torch.float32 )

        for i in range( self.n_blocks ):
            
            n0_matches[i] = 2**i
            
            for j in range( n0_matches[i], 0, -1 ):
                n1_info_size[i] += j
                
            n2_info_size_with_channels_raw[i] = self.init_channel_count * n1_info_size[i]
            
            n3_info_size_with_channels_processed[i] = n2_info_size_with_channels_raw[i] * (self.ch_expans_param**i)
            
            n4_info_size_with_channels_processed_reduced[i] = n3_info_size_with_channels_processed[i] * self.channel_reduction_ratio
        
        return n3_info_size_with_channels_processed.to(torch.int), n4_info_size_with_channels_processed_reduced.to(torch.int)
    
    def forward(self, x):
        
        x = self.net( x )
        
        x = x.view( -1, self.initial_fully_connected_size * 1 * 1 )
        x = self.non_lin( self.fc1(x) )
        x = self.fc2(x)
        
        return x
        
def get_model( config, N, model_width, en_checkpointing, model_adjust_params = None ):   

    if( N == 512 or N == 1024 or N == 2048 ):
        
        in_width = model_width
        
        # non_lin = 'ReLU'
        non_lin = 'LeakyReLU'  
        
        if( config.model_exp_no < 1000 ):
            enable_context_norm = False
        else:
            enable_context_norm = True
        
        if(model_adjust_params != None):            
            init_channel_count = model_adjust_params[0]
            ch_expans_param = model_adjust_params[1]
            height_reduction_type = model_adjust_params[2]
            channel_reduction_layer_enable = model_adjust_params[3]
            channel_reduction_ratio = model_adjust_params[4]
            pointwise_conv_layers_count = model_adjust_params[5]
            pointwise_conv_layers_param = model_adjust_params[6]            
        else:        
            if( (config.model_exp_no >= 0 and config.model_exp_no < 10 ) or (config.model_exp_no >= 1000 and config.model_exp_no < 1010 ) ):
                    
                channel_reduction_ratio = 1
                pointwise_conv_layers_count = 0
                pointwise_conv_layers_param = 0
                
                height_reduction_type='normal_conv'                
                channel_reduction_layer_enable = 0
                
                if( config.model_exp_no < 1000 ):
                    exp_no_list_index = config.model_exp_no - 0
                else:
                    exp_no_list_index = config.model_exp_no - 1000
                
                init_channel_counts = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
                ch_expans_params = [ [ 0.46469, 0.42685, 0.40584, 0.39142, 0.37167, 0.35800, 0.33913, 0.32585, 0.30707, 0.29341, ],
                                     [ 0.45120, 0.41820, 0.39980, 0.38713, 0.36974, 0.35771, 0.34107, 0.32943, 0.31313, 0.30144, ],
                                     [ 0.44057, 0.41133, 0.39496, 0.38367, 0.36815, 0.35737, 0.34252, 0.33214, 0.31766, 0.30739, ], ]

                init_channel_count = init_channel_counts[exp_no_list_index]
                ch_expans_param = ch_expans_params[ int( math.log2(N) - math.log2(512) ) ][exp_no_list_index]
                    
            elif( (config.model_exp_no >= 100 and config.model_exp_no < 110 ) or (config.model_exp_no >= 1100 and config.model_exp_no < 1110 ) ):
                    
                channel_reduction_ratio = 1
                pointwise_conv_layers_count = 0
                pointwise_conv_layers_param = 0
                
                height_reduction_type='depth_wise_sep_conv'
                channel_reduction_layer_enable = 0
                
                if( config.model_exp_no < 1100 ):
                    exp_no_list_index = config.model_exp_no - 100
                else:
                    exp_no_list_index = config.model_exp_no - 1100
                
                init_channel_counts = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
                ch_expans_params = [ [ 0.48052, 0.44197, 0.42066, 0.40602, 0.38603, 0.37223, 0.35327, 0.34009, 0.32165, 0.30855, ],
                                     [ 0.46509, 0.43156, 0.41290, 0.40006, 0.38247, 0.37029, 0.35356, 0.34195, 0.32576, 0.31433, ],
                                     [ 0.45295, 0.42329, 0.40671, 0.39526, 0.37955, 0.36868, 0.35371, 0.34329, 0.32885, 0.31869, ], ]

                init_channel_count = init_channel_counts[exp_no_list_index]
                ch_expans_param = ch_expans_params[ int( math.log2(N) - math.log2(512) ) ][exp_no_list_index]
                
            elif( (config.model_exp_no >= 200 and config.model_exp_no < 205 ) or (config.model_exp_no >= 1200 and config.model_exp_no < 1205 ) ):
                    
                channel_reduction_ratios = [0.6, 0.7, 0.8, 0.9, 1.0]
                pointwise_conv_layers_count = 0
                pointwise_conv_layers_param = 0
                
                height_reduction_type='normal_conv'                
                channel_reduction_layer_enable = 1
                
                if( config.model_exp_no < 1200 ):
                    exp_no_list_index = config.model_exp_no - 200
                else:
                    exp_no_list_index = config.model_exp_no - 1200
                
                init_channel_count = 32
                ch_expans_params = [ [ 0.37491, 0.37111, 0.36784, 0.36498, 0.36240, ],     ### Need to be modified
                                     [ 0.37260, 0.36927, 0.36639, 0.36385, 0.36159, ],     ### Need to be modified
                                     [ 0.37072, 0.36774, 0.36515, 0.36289, 0.36086, ], ]   ### Need to be modified

                channel_reduction_ratio = channel_reduction_ratios[exp_no_list_index]
                ch_expans_param = ch_expans_params[ int( math.log2(N) - math.log2(512) ) ][exp_no_list_index]
                
            elif( (config.model_exp_no >= 300 and config.model_exp_no < 305 ) or (config.model_exp_no >= 1300 and config.model_exp_no < 1305 ) ):
                    
                channel_reduction_ratios = [0.6, 0.7, 0.8, 0.9, 1.0]
                pointwise_conv_layers_count = 0
                pointwise_conv_layers_param = 0
                
                height_reduction_type='depth_wise_sep_conv'
                channel_reduction_layer_enable = 1
                
                if( config.model_exp_no < 1300 ):
                    exp_no_list_index = config.model_exp_no - 300
                else:
                    exp_no_list_index = config.model_exp_no - 1300
                
                init_channel_count = 32
                ch_expans_params = [ [ 0.38246, 0.37861, 0.37530, 0.37237, 0.36973, ],     ### Need to be modified
                                     [ 0.37932, 0.37593, 0.37301, 0.37041, 0.36810, ],     ### Need to be modified
                                     [ 0.37676, 0.37374, 0.37111, 0.36879, 0.36671, ], ]   ### Need to be modified

                channel_reduction_ratio = channel_reduction_ratios[exp_no_list_index]
                ch_expans_param = ch_expans_params[ int( math.log2(N) - math.log2(512) ) ][exp_no_list_index]
                
            elif( (config.model_exp_no >= 400 and config.model_exp_no < 410 ) or (config.model_exp_no >= 1400 and config.model_exp_no < 1410 ) ):
                    
                channel_reduction_ratio = 1
                pointwise_conv_layers_count = int( math.log2(N) )
                pointwise_conv_layers_param = 0
                
                height_reduction_type='normal_conv'                
                channel_reduction_layer_enable = 0
                
                if( config.model_exp_no < 1400 ):
                    exp_no_list_index = config.model_exp_no - 400
                else:
                    exp_no_list_index = config.model_exp_no - 1400
                
                init_channel_counts = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
                ch_expans_params = [ [ 0.46469, 0.42685, 0.40584, 0.39142, 0.37167, 0.35800, 0.33913, 0.32585, 0.30707, 0.29341, ],     ### Need to be modified
                                     [ 0.45120, 0.41820, 0.39980, 0.38713, 0.36974, 0.35771, 0.34107, 0.32943, 0.31313, 0.30144, ],     ### Need to be modified
                                     [ 0.44057, 0.41133, 0.39496, 0.38367, 0.36815, 0.35737, 0.34252, 0.33214, 0.31766, 0.30739, ], ]   ### Need to be modified

                init_channel_count = init_channel_counts[exp_no_list_index]
                ch_expans_param = ch_expans_params[ int( math.log2(N) - math.log2(512) ) ][exp_no_list_index]
                
            elif( (config.model_exp_no >= 500 and config.model_exp_no < 510 ) or (config.model_exp_no >= 1500 and config.model_exp_no < 1510 ) ):
                    
                channel_reduction_ratio = 1
                pointwise_conv_layers_count = int( math.log2(N) )
                pointwise_conv_layers_param = 0
                
                height_reduction_type='depth_wise_sep_conv'
                channel_reduction_layer_enable = 0
                
                if( config.model_exp_no < 1500 ):
                    exp_no_list_index = config.model_exp_no - 500
                else:
                    exp_no_list_index = config.model_exp_no - 1500
                
                init_channel_counts = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
                ch_expans_params = [ [ 0.48052, 0.44197, 0.42066, 0.40602, 0.38603, 0.37223, 0.35327, 0.34009, 0.32165, 0.30855, ],     ### Need to be modified
                                     [ 0.46509, 0.43156, 0.41290, 0.40006, 0.38247, 0.37029, 0.35356, 0.34195, 0.32576, 0.31433, ],     ### Need to be modified
                                     [ 0.45295, 0.42329, 0.40671, 0.39526, 0.37955, 0.36868, 0.35371, 0.34329, 0.32885, 0.31869, ], ]   ### Need to be modified

                init_channel_count = init_channel_counts[exp_no_list_index]
                ch_expans_param = ch_expans_params[ int( math.log2(N) - math.log2(512) ) ][exp_no_list_index]
            
            else:
                raise ValueError(f"The provided argument is not valid: {config.model_exp_no}")
                
        return model_exp_00( N, 
                             in_width,
                             init_channel_count,
                             ch_expans_param,
                             height_reduction_type,
                             channel_reduction_layer_enable,
                             channel_reduction_ratio,
                             pointwise_conv_layers_count,
                             pointwise_conv_layers_param,
                             enable_context_norm,
                             non_lin, )
    else:
        raise ValueError(f"The provided argument is not valid: {N}")

if __name__ == '__main__':
    
    import os    
    os.chdir( os.path.dirname( os.getcwd( ) ) )       
    from config import get_config
    from models.models import get_model_structure
    from models.models import get_model_params_and_FLOPS
    config = get_config()
    
    device = 'cpu'
    
    # N = 512
    # N = 1024
    N = 2048
    model_width = 4
    en_checkpointing = False
    
    # first_model_no = 0
    # last_model_no = 10
    
    # first_model_no = 100
    # last_model_no = 110
    
    # first_model_no = 200
    # last_model_no = 205
    
    # first_model_no = 300
    # last_model_no = 305
    
    # first_model_no = 400
    # last_model_no = 410
    
    # first_model_no = 500
    # last_model_no = 510
    
####################################################################################
    
    first_model_no = 1000
    last_model_no = 1010
    
    # first_model_no = 1100
    # last_model_no = 1110
    
    # first_model_no = 1200
    # last_model_no = 1205
    
    # first_model_no = 1300
    # last_model_no = 1305
    
    # first_model_no = 1400
    # last_model_no = 1410
    
    # first_model_no = 1500
    # last_model_no = 1510
    
    for i in range( first_model_no, last_model_no, 1 ):
        config.model_exp_no = i 
        print(f'config.model_exp_no: {config.model_exp_no}')
        model = get_model( config, N, model_width, en_checkpointing ).to(device)
        get_model_structure( config, device, model, N, model_width, en_grad_checkpointing=0)
        # get_model_params_and_FLOPS( config, device, model, N, model_width, en_grad_checkpointing=0)
        print('-'*80)

