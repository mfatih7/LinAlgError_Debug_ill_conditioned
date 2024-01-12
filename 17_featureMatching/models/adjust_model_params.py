import torch

from torchsummary import summary
from thop import profile
from torchSummaryWrapper import get_torchSummaryWrapper

from models_exp import get_model

def get_model_structure( config, device, model, N, model_width, en_grad_checkpointing):   
    
    if(en_grad_checkpointing==False):
        summary(model, (config.input_channel_count, N, model_width), batch_size=2, device=device ) # batch_size must be at least 2 to prevent batch norm errors
    else:                    
        summary(get_torchSummaryWrapper( model ), (config.input_channel_count, N, model_width), batch_size=2, device=device ) # batch_size must be at least 2 to prevent batch norm errors
    input_thop = torch.randn(2, config.input_channel_count, N, model_width, device=device)  # Example input tensor
    flops, params = profile(model, inputs=(input_thop, ))
    flops = int(flops)
    params = int(params)
    print(f"Model FLOPs: {flops:,}")
    print(f"Model Parameters: {params:,}")

def get_model_params_and_FLOPS( config, device, model, N, model_width, en_grad_checkpointing):   
    input_thop = torch.randn(2, config.input_channel_count, N, model_width, device=device)  # Example input tensor
    flops, params = profile(model, inputs=(input_thop, ))
    flops = int(flops)
    params = int(params)
    print(f"Model FLOPs: {flops:,}")
    print(f"Model Parameters: {params:,}")
    
    return params, flops
        
def adjust_model_params( config, N, model_width, en_checkpointing, 
                         adjusting_mode, adjusting_param, adjusting_param_value, model_adjust_params ):
    """    
    adjusting_mode == 0 Models params or FLOPS are set by increasing ch_expans_param

    """
    
    if(adjusting_param=='model_params' or adjusting_param=='FLOPS'):
    
        if(adjusting_mode==0): # 
            
            n_digits = 5
            increment = 1 * 10**(-1)
            
            while True:
                for n in range(n_digits):
                    while True:
                        model = get_model( config, N, model_width, en_checkpointing, model_adjust_params ).to(device)
                        params, flops = get_model_params_and_FLOPS( config, device, model, N, model_width, en_grad_checkpointing=0)
                        
                        if( (adjusting_param=='model_params' and adjusting_param_value<params) or
                            (adjusting_param=='FLOPS' and adjusting_param_value<flops) ):
                            if(n<n_digits-1):
                                model_adjust_params[1] -= increment
                                increment = increment / 10
                                model_adjust_params[1] += increment
                            break
                        else:
                            model_adjust_params[1] += increment
                if(n==n_digits-1):
                    break
                    
            if(adjusting_param=='model_params' and adjusting_param_value<params):
                print(f'Adjusted value is {model_adjust_params[1]} for model_params')
                return params, model_adjust_params[1]
            elif(adjusting_param=='FLOPS' and adjusting_param_value<flops):
                print(f'Adjusted value is {model_adjust_params[1]} for FLOPS')
                return flops, model_adjust_params[1]          
        else:
            raise ValueError(f"The provided argument is not valid: {adjusting_mode}")
    else:
        raise ValueError(f"The provided argument is not valid: {adjusting_param}")

if __name__ == '__main__':
    
    import os    
    os.chdir( os.path.dirname( os.getcwd( ) ) )       
    from config import get_config
    config = get_config()
    
    device = 'cpu'
    
    # N = 512
    # N = 1024
    N = 2048
    model_width = 4
    en_checkpointing = False
    
##############################################################################################
    
    # N_experiments = 10
    
    # config.model_exp_no = 0
    # # config.model_exp_no = 100

    # init_channel_counts = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    # ch_expans_param = 0.30000
    # if(config.model_exp_no >= 0 and config.model_exp_no < 10 ):        
    #     height_reduction_type = 'normal_conv'        
    # elif(config.model_exp_no >= 100 and config.model_exp_no < 110 ):        
    #     height_reduction_type = 'depth_wise_sep_conv'
    # channel_reduction_layer_enable = 0
    # channel_reduction_ratio = 1
    # pointwise_conv_layers_count = 0
    # pointwise_conv_layers_param = 0    
    
    # adjusting_mode = 0
    # adjusting_param = 'model_params'
    
    # adjusting_param_value = 2_000_000 * ( N/512 )
    
    # model_adjust_params = []
    
    # model_adjust_params.append(0)
    # model_adjust_params.append(ch_expans_param)
    # model_adjust_params.append(height_reduction_type)
    # model_adjust_params.append(channel_reduction_layer_enable)
    # model_adjust_params.append(channel_reduction_ratio)
    # model_adjust_params.append(pointwise_conv_layers_count)
    # model_adjust_params.append(pointwise_conv_layers_param)
    
    # adjusted_values = []
    # adjusted_params = []

    # for i in range( N_experiments ):
        
    #     model_adjust_params[0] = init_channel_counts[i]
        
    #     adjusted_value, adjusted_param = adjust_model_params( config, N, model_width, en_checkpointing, 
    #                                                           adjusting_mode, adjusting_param, adjusting_param_value, model_adjust_params )
    #     adjusted_values.append(adjusted_value)
    #     adjusted_params.append(adjusted_param)
        
##############################################################################################

    N_experiments = 5

    # config.model_exp_no = 200
    config.model_exp_no = 300
    
    init_channel_count = 64
    ch_expans_param = 0.30000
    if(config.model_exp_no >= 200 and config.model_exp_no < 205 ):        
        height_reduction_type = 'normal_conv'        
    elif(config.model_exp_no >= 300 and config.model_exp_no < 305 ):        
        height_reduction_type = 'depth_wise_sep_conv'
    channel_reduction_layer_enable = 1
    channel_reduction_ratios = [0.6, 0.7, 0.8, 0.9, 1.0]
    pointwise_conv_layers_count = 0
    pointwise_conv_layers_param = 0    
    
    adjusting_mode = 0
    adjusting_param = 'model_params'
    
    adjusting_param_value = 4_000_000 * ( N/512 )
    
    model_adjust_params = []
        
    model_adjust_params.append(init_channel_count)
    model_adjust_params.append(ch_expans_param)
    model_adjust_params.append(height_reduction_type)
    model_adjust_params.append(channel_reduction_layer_enable)
    model_adjust_params.append(0)
    model_adjust_params.append(pointwise_conv_layers_count)
    model_adjust_params.append(pointwise_conv_layers_param)
    
    adjusted_values = []
    adjusted_params = []

    for i in range( N_experiments ):
        
        model_adjust_params[4] = channel_reduction_ratios[i]
        
        adjusted_value, adjusted_param = adjust_model_params( config, N, model_width, en_checkpointing, 
                                                              adjusting_mode, adjusting_param, adjusting_param_value, model_adjust_params )
        adjusted_values.append(adjusted_value)
        adjusted_params.append(adjusted_param) 
