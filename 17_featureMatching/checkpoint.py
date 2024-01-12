import os
import shutil
import numpy as np
import torch

# import torch_xla.core.xla_model as xm
# import torch_xla.utils.serialization as xser

def save_checkpoints_cpu_gpu_tpu( config, checkpoint, path_checkpoints_file ):
    
    if(config.device != 'tpu' ):
        torch.save(checkpoint, path_checkpoints_file )
    else:
        import torch_xla.core.xla_model as xm
        xm.save(checkpoint, path_checkpoints_file, master_only=True ) # Default master_only True
        # xser.save(checkpoint, path_checkpoints_file, master_only=True ) # Default master_only True

def save_initial_checkpoint( config, model, optimizer ):

    checkpoint_path = os.path.join( config.output_path_local, 'checkpoints' )
    if(os.path.exists(checkpoint_path)==0):

        if(config.device != 'tpu' ):
            os.makedirs(checkpoint_path)
        else:
            import torch_xla.core.xla_model as xm
            if(xm.is_master_ordinal()):
                os.makedirs(checkpoint_path)
        
        success_checkpoint = np.zeros( (2, config.n_epochs[0], config.n_chunks, 4) )
        loss_checkpoint = np.zeros( (2, config.n_epochs[0], config.n_chunks, 3) )
        proc_time_checkpoint = np.zeros( (2, config.n_epochs[0], config.n_chunks) )
        
        checkpoint = {
                      'epoch' : 0,
                      'chunk' : 0,
                      
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                      # 'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                      
                      'success_checkpoint': success_checkpoint,
                      'loss_checkpoint': loss_checkpoint,
                      'proc_time_checkpoint': proc_time_checkpoint,
                     }
        checkpoint_file_with_path = os.path.join(checkpoint_path, 'model.pth.tar')
        
        # torch.save(checkpoint, checkpoint_file_with_path )
        save_checkpoints_cpu_gpu_tpu( config, checkpoint, checkpoint_file_with_path )

def load_checkpoint( config, device, model, optimizer ):
    
    checkpoint_path = os.path.join( config.output_path_local, 'checkpoints' )
    checkpoint_file_with_path = os.path.join(checkpoint_path, 'model.pth.tar')

    if(config.device == 'tpu' ):
        import torch_xla.core.xla_model as xm
        import torch_xla.utils.serialization as xser
        
        print( f"{xm.xla_real_devices([str(device)])[0]} DEB PNT 4A" )
    
    # checkpoint = torch.load(checkpoint_file_with_path)
    if(config.device != 'tpu' ):
        checkpoint = torch.load( checkpoint_file_with_path )
    else:        
        # import torch_xla.utils.serialization as xser
        print( f"{xm.xla_real_devices([str(device)])[0]} DEB PNT 4AA {checkpoint_file_with_path}" )
        print( f"{xm.xla_real_devices([str(device)])[0]} DEB PNT 4AAA {os.getcwd()}" )
        while True:
            if( os.path.exists(checkpoint_file_with_path) ):
                print( f"{xm.xla_real_devices([str(device)])[0]} DEB PNT 4AAAA {checkpoint_file_with_path}" )
                checkpoint = xser.load( checkpoint_file_with_path )
                break
    if(config.device == 'tpu' ):
        print( f"{xm.xla_real_devices([str(device)])[0]} DEB PNT 4B" )
    
    epoch = checkpoint['epoch']
    chunk = checkpoint['chunk']
    
    if(config.device == 'tpu' ):
        print( f"{xm.xla_real_devices([str(device)])[0]} DEB PNT 4C" )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if(config.device == 'tpu' ):
        print( f"{xm.xla_real_devices([str(device)])[0]} DEB PNT 4D" )

    model.to(device)
    
    if(config.device == 'tpu' ):
        print( f"{xm.xla_real_devices([str(device)])[0]} DEB PNT 4E" )
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    success_checkpoint = checkpoint['success_checkpoint']
    loss_checkpoint = checkpoint['loss_checkpoint']
    proc_time_checkpoint = checkpoint['proc_time_checkpoint']
    
    if(config.device == 'tpu' ):
        print( f"{xm.xla_real_devices([str(device)])[0]} DEB PNT 4F" )
    
    return epoch, chunk, model, optimizer, success_checkpoint, loss_checkpoint, proc_time_checkpoint

def save_checkpoint( config, epoch, chunk, model, optimizer, success_checkpoint, loss_checkpoint, proc_time_checkpoint):
    
    checkpoint_path = os.path.join( config.output_path_local, 'checkpoints' )
    checkpoint_file_with_path = os.path.join(checkpoint_path, 'model.pth.tar')
    if(config.save_checkpoint_last_or_all == 'all'):
        archive_checkpoint_file_with_path = os.path.join(checkpoint_path, 'model' + f'_{epoch:04d}_{chunk:04d}' + '.pth.tar')    
    elif(config.save_checkpoint_last_or_all == 'last'):
        archive_checkpoint_file_with_path = os.path.join(checkpoint_path, 'model' + '_prev' + '.pth.tar')   
    shutil.copyfile(checkpoint_file_with_path, archive_checkpoint_file_with_path) 
    
    if(chunk==config.n_chunks-1):
        chunk = 0
        epoch = epoch + 1
    else:
        chunk = chunk + 1
    
    checkpoint = {
                  'epoch' : epoch,
                  'chunk' : chunk,
                  
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  # 'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                  
                  'success_checkpoint': success_checkpoint,
                  'loss_checkpoint': loss_checkpoint,
                  'proc_time_checkpoint': proc_time_checkpoint,
                 }       
    # torch.save(checkpoint, checkpoint_file_with_path )
    save_checkpoints_cpu_gpu_tpu( config, checkpoint, checkpoint_file_with_path )

# Checkpoint Functions For Test

def load_test_checkpoint( config, device, model, checkpoint_file_with_path):
    
    # if( os.path.exists( os.path.join(config.output_path_local, 'checkpoints', 'model.pth.tar') ) ):
    #     checkpoint_file_with_path = os.path.join(config.output_path_local, 'checkpoints', 'model.pth.tar')
    # else:
    #     assert(0)

    if(config.device=='tpu'):
        import torch_xla.utils.serialization as xser
        checkpoint = xser.load( checkpoint_file_with_path )
    else:
        checkpoint = torch.load(checkpoint_file_with_path)
    
    print( 'Loading checkpoint file ' + checkpoint_file_with_path )
    
    model.load_state_dict(checkpoint['model_state_dict'])    
    model.to(device)
    
    return model

def update_mAP_checkpoint(mAP_checkpoint, err_q, err_t, err_qt, epoch, chunk):
    
    if(mAP_checkpoint.shape[0]==2):        
        assert(0)        
    else:        
        for err_ind, error in enumerate( [err_q, err_t, err_qt] ):            
            mAP_checkpoint[ 0, epoch, chunk, err_ind, int(np.ceil(error)-1) ] += 1    
    return mAP_checkpoint

def get_all_checkpoint_files_for_test(config):
    checkpoint_files = []
    
    for filename in os.listdir( os.path.join( config.output_path_local, 'checkpoints' ) ):
        filename_with_path = os.path.join(config.output_path_local, 'checkpoints', filename)
        if os.path.isfile(filename_with_path):
            checkpoint_files.append(filename_with_path)
    
    #Sorting needed for linux operation
    checkpoint_files.sort()

    # Remove Checkpoint of Epoch 0, Chunk 0 since it is the initial checkpoint saved at the start of Epoch 0, Chunk 0    
    elements_to_change = [ os.path.join( config.output_path_local, 'checkpoints', 'model' + f'_{0:04d}_{0:04d}' + '.pth.tar'),\
                           os.path.join( config.output_path_local, 'checkpoints', 'model_prev.pth.tar'),\
                           os.path.join( config.output_path_local, 'checkpoints', 'model.pth.tar') ]
    for element_to_change_id, element_to_change in enumerate(elements_to_change):
        if element_to_change in checkpoint_files:
            checkpoint_files.remove(element_to_change)
            if(element_to_change_id>0):
                checkpoint_files.append(element_to_change)
            
    return checkpoint_files
    
def save_test_checkpoint( config, success_checkpoint, loss_checkpoint, proc_time_checkpoint, mAP_checkpoint):
    
    checkpoint_path = os.path.join( config.output_path_local, 'checkpoints_test' )
    if(os.path.exists(checkpoint_path)==0):
        os.makedirs(checkpoint_path)
    
    checkpoint_file_with_path = os.path.join(checkpoint_path, 'checkpoint_test.pth.tar')    
    
    checkpoint = {                  
                  'success_checkpoint': success_checkpoint,
                  'loss_checkpoint': loss_checkpoint,
                  'proc_time_checkpoint': proc_time_checkpoint,
                  'mAP_checkpoint': mAP_checkpoint,
                 }       
    # torch.save(checkpoint, checkpoint_file_with_path )
    save_checkpoints_cpu_gpu_tpu( config, checkpoint, checkpoint_file_with_path )

def find_best_test_checkpoint( config, ref_angles):
    
    checkpoint_path = os.path.join( config.output_path_local, 'checkpoints_test' )
    
    checkpoint_file_with_path = os.path.join(checkpoint_path, 'checkpoint_test.pth.tar')
    
    if(config.device=='tpu'):
        import torch_xla.utils.serialization as xser
        checkpoint = xser.load( checkpoint_file_with_path )
    else:
        checkpoint = torch.load(checkpoint_file_with_path)
    
    success_checkpoint = checkpoint['success_checkpoint']
    loss_checkpoint = checkpoint['loss_checkpoint']
    proc_time_checkpoint = checkpoint['proc_time_checkpoint']
    mAP_checkpoint = checkpoint['mAP_checkpoint']
    
    x_count = mAP_checkpoint.shape[1] * mAP_checkpoint.shape[2]
    
    error_angle_legend_names = ['Rt']    
    err_ang_ind = 2
    
    mAP = np.zeros( (1, len(ref_angles), x_count) )
    loss = np.zeros( (1, 3, x_count) )    
    
    for ref_angle_ind, ref_angle in enumerate(ref_angles):
        for e in range(mAP_checkpoint.shape[1]):
            for c in range(mAP_checkpoint.shape[2]):
                mAP[0, ref_angle_ind, e*mAP_checkpoint.shape[2]+c] = np.sum(mAP_checkpoint[0, e, c, err_ang_ind, 0:ref_angle]) / np.sum(mAP_checkpoint[0, e, c, err_ang_ind, :]) * 100
                
                if(ref_angle_ind == 0):
                    loss[0, :, e*mAP_checkpoint.shape[2]+c] = loss_checkpoint[0, e, c, :]
    
    best_exp_id = np.argmax(mAP[0,0,:])
    
    print( f'Best experiment mAP 5 is {mAP[0,0,best_exp_id]} ')
    print( f'Best experiment mAP 10 is {mAP[0,1,best_exp_id]} ')
    print( f'Best experiment mAP 20 is {mAP[0,2,best_exp_id]} ')
    
    print( f'Best experiment loss cls is {loss[0,0,best_exp_id]} ')
    print( f'Best experiment loss geo is {loss[0,1,best_exp_id]} ')
    print( f'Best experiment loss ess is {loss[0,2,best_exp_id]} ')
    
    print( f'Best experiment is the {best_exp_id}th experiment ')
    
    
    
    