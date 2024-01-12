from config import get_config

from process.train_1_1_each_sample_in_multi_batch import train_and_val as train_1_1_each_sample_in_multi_batch_train_and_val
from process.train_1_1_each_sample_in_single_batch import train_and_val as train_1_1_each_sample_in_single_batch_train_and_val

from process.train_n_n import train_and_val as train_n_n_train_and_val

# Wrap most of you main script’s code within if __name__ == '__main__': block, to make sure it doesn’t run again
# (most likely generating error) when each worker process is launched. You can place your dataset and DataLoader
# instance creation logic here, as it doesn’t need to be re-executed in workers.

if __name__ == '__main__':

    config = get_config()
    
    experiment = config.first_experiment
    
    config.copy_config_file_to_output_folder(experiment)
    
    learning_rate = config.learning_rate
    n_epochs = config.n_epochs
    num_workers = config.num_workers
    model_type = config.model_type
    optimizer_type = config.optimizer_types[0]
    en_grad_checkpointing = config.en_grad_checkpointing
    N_images_in_batch = config.training_params[0][0]
    N = config.training_params[0][1]
    batch_size = config.training_params[0][2]
    
    if(config.operation == 'train'):
    
        if(config.input_type == '1_to_1'):
        
            if( N_images_in_batch == 1 and N > batch_size ):
                
                print('Training starts for ' + 'train_1_1_each_sample_in_multi_batch')            
                training_results = train_1_1_each_sample_in_multi_batch_train_and_val(   
                                                                                        config,
                                                                                        learning_rate,
                                                                                        n_epochs,
                                                                                        num_workers,
                                                                                        model_type,
                                                                                        en_grad_checkpointing,
                                                                                        N_images_in_batch,
                                                                                        N,
                                                                                        batch_size,
                                                                                        optimizer_type, )
                    
            elif( N_images_in_batch >= 1 and N == batch_size ):
                
                print('Training starts for ' + 'train_1_1_each_sample_in_single_batch')
                training_results = train_1_1_each_sample_in_single_batch_train_and_val(   
                                                                                        config,
                                                                                        learning_rate,
                                                                                        n_epochs,
                                                                                        num_workers,
                                                                                        model_type,
                                                                                        en_grad_checkpointing,
                                                                                        N_images_in_batch,
                                                                                        N,
                                                                                        batch_size,
                                                                                        optimizer_type, )            
            else:
                raise ValueError(f"The provided arguments are not valid: {N_images_in_batch} {N} {batch_size}")
                
        elif(config.input_type == 'n_to_n'):
            
            if( N_images_in_batch == batch_size  ):
            
                print('Training starts for ' + 'train_n_n')            
                training_results = train_n_n_train_and_val(   
                                                            config,
                                                            learning_rate,
                                                            n_epochs,
                                                            num_workers,
                                                            model_type,
                                                            en_grad_checkpointing,
                                                            N_images_in_batch,
                                                            N,
                                                            batch_size,
                                                            optimizer_type, )
            else:
                raise ValueError(f"The provided arguments are not valid: {N_images_in_batch} {batch_size}")
        else:
            raise ValueError(f"The provided arguments are not valid: {config.input_type}")
    else:
        raise ValueError(f"The provided arguments are not valid: {config.operation}")