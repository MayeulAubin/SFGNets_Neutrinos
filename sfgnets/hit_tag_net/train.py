################ SCRIPT FOR TRAINING ####################

import argparse
    
parser = argparse.ArgumentParser(
                    prog='HitTagTraining',
                    description='Trains a model for Hit Tagging in SFG',)

parser.add_argument('j', metavar='j', type=int, help='#j div of the model')
parser.add_argument('dataset_folder',metavar='Dataset_Folder', type=str, help="Folder in which are stored the event_#.npz files for training")
parser.add_argument('scaler_file',metavar='Scaler_File', type=str, help="File storing the dataset features scalers")
parser.add_argument('save_path',metavar='Save_Path', type=str, help="Path to save results and models")
parser.add_argument('-T', '--test', action='store_true', help='runs test after training (classification report,...)')
parser.add_argument('--test_only', action='store_true', help='runs only the test (classification report,...)')
parser.add_argument('-m', '--multi_GPU', action='store_true', help='runs the script on multi GPU')
parser.add_argument('-b', '--benchmarking', action='store_true', help='prints the duration of the different parts of the code')
parser.add_argument('-s', '--sub_tqdm', action='store_true', help='displays the progress bars of the train and test loops for each epoch')
parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
parser.add_argument('-n','--num_epochs', type=int, default=200, help='number of epochs for the training')
parser.add_argument('--multi_pass', type=int, default=1, help='how many times the whole dataset is gone through in an epoch')
parser.add_argument('--lr', type=float, default=1e-3, help='maximum learning rate for training (defines the scale of the learning rate)')
parser.add_argument('--stop_after_epochs', type=int, default=40, help='maximum number of epochs without improvement before stopping the training (early termination)')
parser.add_argument('-f', '--filter', action='store_true', help='filter the hits to keep only the ones close to the vertex, use --cut to control the distance of the cut')
parser.add_argument('--cut', type=float, default=8., help='when filtering, maximum distance of the hits to the vertex to be kept')
parser.add_argument('-r', '--retag', action='store_true', help='retag the hit tags according to the retag_cut function, use --rcut to control the distance of the cut')
parser.add_argument('--rcut', type=float, default=40., help='when filtering, maximum distance of the hits to the vertex to be kept')
parser.add_argument("-G", "--gpu", type=int, default=1, help="GPU ID (cuda) to be used")

import os
import pickle as pk
import numpy as np
import time
import MinkowskiEngine as ME
from sklearn.metrics import  classification_report

from .model import *
from .losses import *
from .execution import training, test_full
from .dataset import collate_sparse_minkowski, SparseEvent, retag_cut, select_hits_near_vertex
from ..datasets.utils import full_dataset



def ddp_setup(rank, world_size):
    """
    Initialise the DDP for multi_GPU application
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main_worker(device,
                dataset,
                args,
                world_size=1,
                multi_GPU=False):

    #### Get the variables from args
    ## Get the first positional argument passed to the script (the j div of the training)
    j=args.j

    ## Get the multi_GPU flag
    multi_GPU=args.multi_GPU

    ## Get the benchmarking flag
    benchmarking=args.benchmarking
    sub_progress_bars=args.sub_tqdm

    multi_pass=args.multi_pass
            
    
    print(f"Starting main worker on device {device}")
    
    # if we are in a multi_GPU training setup, we need to initialise the DDP (Distributed Data Parallel)
    if multi_GPU:
        ddp_setup(rank=device, world_size=world_size)
    
    
    model = minkunet.MinkUNet34B(in_channels=4, out_channels=3, D=3).to(device)
    print(model.__str__().split('\n')[0][:-1]) # Print the model name
    
    if multi_GPU:
        model=torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[device])
        # For multi-GPU training, DDP requires to change BatchNorm layers to SyncBatchNorm layers
        model=ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    
    # Print the total number of trainable parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {pytorch_total_params}")
        
    # Loss function
    loss_func= SumLoss([FocalLoss, DiceLoss(num_classes=3)])

    # Loaders parameters
    batch_size=args.batch_size
    train_fraction=0.9
    val_fraction=0.099
    seed=7
    len_train_loader=int(len(dataset)*train_fraction/(multi_pass*batch_size))

    # Optimizer and scheduler
    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-9, weight_decay=0.01)
    num_steps_one_cycle = 25
    num_warmup_steps = 10
    cosine_annealing_steps = len_train_loader * num_steps_one_cycle
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cosine_annealing_steps, T_mult=1, eta_min=lr*(1e-2))
    warmup_scheduler = WarmUpScheduler(optimizer, lr_scheduler,
                                    len_loader=1,
                                    warmup_steps=len_train_loader * num_warmup_steps,
                                    warmup_start_lr=lr/100,
                                    warmup_mode='linear')

    num_epochs=args.num_epochs
    stop_after_epochs=args.stop_after_epochs
    
    training_dict=training(device=device,
                            model=model,
                            dataset=dataset,
                            optimizer=optimizer,
                            warmup_scheduler=warmup_scheduler,
                            loss_func=loss_func,
                            batch_size=batch_size,
                            train_fraction=train_fraction,
                            val_fraction=val_fraction,
                            seed=seed,
                            epochs=num_epochs,
                            stop_after_epochs= stop_after_epochs,
                            progress_bar=True,
                            benchmarking=benchmarking,
                            sub_progress_bars=sub_progress_bars,
                            multi_GPU=multi_GPU,
                            world_size=world_size,
                            save_model_path=f"{args.save_path}models/hittag_model_{j}.torch")
    
    if multi_GPU:
        torch.distributed.destroy_process_group()
    
    return model, training_dict



def _test_model(device,
                dataset,
                args,
                model):
    
    full_loader = full_dataset(dataset,
                                collate=collate_sparse_minkowski)
    all_results = test_full(full_loader,
                            model,
                            progress_bar=True,
                            )

    print("Saving results...")
    with open(f'{args.save_path}results/hit_tagging_model_{args.j}_pred.pkl', 'wb') as file:
        pk.dump(all_results,file)
    
    print("Flattening and extracting...")
    ## Get a flat array of true and predicted labels
    target_names = ["Vertex activity", "Single P", "Noise"]
    val_true=np.vstack(all_results['y']).flatten()

    y_pred=np.vstack(all_results['predictions'])
    val_pred=np.argmax(y_pred,axis=-1) # argmax solution
    # val_pred=torch.multinomial(torch.Tensor(y_pred).to(device),num_samples=1)[...,0].cpu().numpy() # random choice solution
    
    print("Printing classification report...")
    print(classification_report(val_true, val_pred, digits=3, target_names=target_names))
    
    
    
    




if __name__ == "__main__":
    
    args = parser.parse_args()

    ## Get the first positional argument passed to the script (the j div of the training)
    j=args.j

    ## Get the multi_GPU flag
    multi_GPU=args.multi_GPU

    ## Get the benchmarking flag
    benchmarking=args.benchmarking
    sub_progress_bars=args.sub_tqdm

    multi_pass=args.multi_pass

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    torch.multiprocessing.set_sharing_strategy('file_system')
    
    
    print(args.dataset_folder[:-6]+"test/")



    def main():
        global multi_GPU, benchmarking
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if (not args.test_only):
        
            if multi_GPU:
                print(f"Training on Multi GPU...")
            else:
                print(f"Training on Single GPU {device}...")
            if benchmarking:
                print("Benchmarking...")
            if multi_pass!=1:
                print(f"Multi pass {multi_pass}...")

            # generate dataset
            dataset=SparseEvent(args.dataset_folder,
                                scaler_file=args.scaler_file,
                                multi_pass=multi_pass,
                                filtering_func=select_hits_near_vertex(cut=args.cut,dist_type="cube") if args.filter else None,
                                center_event=args.filter,
                                retagging_func=retag_cut(args.rcut) if args.retag else None) 

            t0=time.perf_counter()

            if multi_GPU:
                world_size = torch.cuda.device_count()
                model, training_dict=torch.multiprocessing.spawn(main_worker, args=(
                                                                        dataset,
                                                                        args,
                                                                        world_size,
                                                                        multi_GPU), nprocs=world_size)
                
            else:
                model, training_dict=main_worker(device,
                                    dataset,
                                    args,)
                
            t0=t0-time.perf_counter()

            torch.save(training_dict,f"{args.save_path}results/training_dict_{j}.torch")    

        ## Runs the tests
        if args.test or args.test_only:
            testdataset_folder=args.dataset_folder[:-6]+"test/" ## we are assuming that the dataset_folder is of type "*_train/" whereas the testdataset folder will be "*_test/"
            testdataset=SparseEvent(testdataset_folder,
                            scaler_file=args.scaler_file,
                            multi_pass=multi_pass,
                            filtering_func=select_hits_near_vertex(cut=args.cut,dist_type="cube") if args.filter else None,
                            center_event=args.filter,
                            retagging_func=retag_cut(args.rcut) if args.retag else None) 
            
            model = minkunet.MinkUNet34B(in_channels=4, out_channels=3, D=3).to(device)
            model.load_state_dict(torch.load(f"{args.save_path}models/hittag_model_{j}.torch"))
            
            _test_model(device=device,
                        dataset=testdataset,
                        args=args,
                        model=model)
    
        
    main()