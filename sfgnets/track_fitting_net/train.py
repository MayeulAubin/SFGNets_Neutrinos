import argparse
import torch
import os
import time
import MinkowskiEngine as ME
from warmup_scheduler_pytorch import WarmUpScheduler

from .dataset import PGunEvent, full_dataset, collate_minkowski, collate_transformer
from .model import create_sparse_cnn_model, create_transformer_model
from .losses import SumPerf, MomentumLoss, loss_fn
from .execution import ddp_setup, training, measure_performances, test_full
from .plot import plots



    
################ SCRIPT FOR TRAINING ####################
## it allows to run track_fitting_net as a script directly   


parser = argparse.ArgumentParser(
                    prog='TrackFittingTraining',
                    description='Trains a model for Track Fitting in SFG',)

parser.add_argument('j', metavar='j', type=int, help='#j div of the model')
parser.add_argument('dataset_folder',metavar='Dataset_Folder', type=str, help="Folder in which are stored the event_#.npz files for training")
parser.add_argument('scaler_file',metavar='Scaler_File', type=str, help="File storing the dataset features scalers")
parser.add_argument('save_path',metavar='Save_Path', type=str, help="Path to save results and models")
parser.add_argument('--test_only', action='store_true', help='runs only the test (measure performances, plots, ...)')
parser.add_argument('-T', '--test', action='store_true', help='runs test after training (measure performances, plots, ...)')
parser.add_argument('-B', '--sparse_cnn', action='store_true', help='use the sparse_cnn model (MinkUNet), otherwise use the transformer')
parser.add_argument('-R', '--resume', action='store_true', help='resume the training by loading the saved model state dictionary')
parser.add_argument('-m', '--multi_GPU', action='store_true', help='runs the script on multi GPU')
parser.add_argument('-b', '--benchmarking', action='store_true', help='prints the duration of the different parts of the code')
parser.add_argument('-s', '--sub_tqdm', action='store_true', help='displays the progress bars of the train and test loops for each epoch')
parser.add_argument('-bs','--batch_size', type=int, default=256, help='batch size for training')
parser.add_argument('-n','--num_epochs', type=int, default=200, help='number of epochs for the training')
parser.add_argument('--multi_pass', type=int, default=1, help='how many times the whole dataset is gone through in an epoch')
parser.add_argument('--lr', type=float, default=1e-3, help='maximum learning rate for training (defines the scale of the learning rate)')
parser.add_argument('--stop_after_epochs', type=int, default=40, help='maximum number of epochs without improvement before stopping the training (early termination)')
parser.add_argument('-w','--weights', type=float, nargs=len(loss_fn), default=loss_fn.weights, help='weights for the loss functions')
parser.add_argument('-t','--targets', type=int, nargs="*", default=None, help='the target indices to include')
parser.add_argument('-i','--inputs', type=int, nargs="*", default=None, help='the input indices to include')
parser.add_argument('--ms', type=str, default='distance', metavar='Masking Scheme', help='Which mask to use for the dataset (distance, primary, tag, ...)')
parser.add_argument('--mom_loss', action='store_true', help='use the momentum loss instead of the simple MSE')
parser.add_argument('--momdir_loss', action='store_true', help='use the momentum-direction loss instead of the simple MSE')  
parser.add_argument('--momsph', action='store_true', help='use spherical coordinates for the momentum')  
parser.add_argument("-G", "--gpu", type=int, default=1, help="GPU ID (cuda) to be used")
    

    
def main_worker(device:torch.device,
                dataset:PGunEvent,
                args:dict,
                world_size:int=1,
                multi_GPU:bool=False):
    global loss_fn
    
    #### Get the variables from args
    ## Get the first positional argument passed to the script (the j div of the training)
    j=args.j

    ## Get the multi_GPU flag
    multi_GPU=args.multi_GPU

    ## Get the benchmarking flag
    benchmarking=args.benchmarking
    sub_progress_bars=args.sub_tqdm

    multi_pass=args.multi_pass
    
    if args.mom_loss:
        # loss_fn.losses[1].loss_func=MomentumLoss()
        loss_fn=loss_fn_mom_loss
    elif args.momdir_loss:
        # loss_fn.losses[1].loss_func=MomDirLoss()
        loss_fn=loss_fn_momdir_loss
    elif args.momsph:
        # loss_fn.losses[1].loss_func=MomSphLoss(dir_weight=1.,norm_weight=1e-2)
        loss_fn=loss_fn_momsph
    
    if args.targets is not None:
        ## Select only the targets for the loss function
        loss_fn=loss_fn[dataset.targets]
        
        ## Replace the weights of the loss function
        loss_fn.weights=[args.weights[k] for k in dataset.targets]
    
    # ## Reconstruct the partial loss indexes, was useful when using one loss per target
    # if args.targets is not None:
    #     loss_fn.rebuild_partial_losses()
    
    ## Get whether we will be using the sparse_cnn model or the transformer
    use_sparse_cnn=args.sparse_cnn
    
    
    print(f"Starting main worker on device {device}")
    
    # if we are in a multi_GPU training setup, we need to initialise the DDP (Distributed Data Parallel)
    if multi_GPU:
        ddp_setup(rank=device, world_size=world_size)
    
    
    if use_sparse_cnn:
        model=create_sparse_cnn_model(y_out_channels=sum(dataset.targets_lengths)+sum(dataset.targets_n_classes), x_in_channels=len(args.inputs) if args.inputs is not None else 2, device=device)
    else:
        model=create_transformer_model(y_out_channels=sum(dataset.targets_lengths)+sum(dataset.targets_n_classes), x_in_channels=len(args.inputs) if args.inputs is not None else 2, device=device)
    
    print(model.__str__().split('\n')[0][:-1]) # Print the model name
    
    if args.resume:
        print("Loading model state dict save...")
        model.load_state_dict(torch.load(f"{args.save_path}models/trackfit_model_{'sparse_cnn_' if use_sparse_cnn else ''}{j}.torch"))
    
    if multi_GPU:
        model=torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[device])
        # For multi-GPU training, DDP requires to change BatchNorm layers to SyncBatchNorm layers
        model=ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model) if use_sparse_cnn else torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # Print the total number of trainable parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {pytorch_total_params}")
    

    # Loaders parameters
    batch_size=args.batch_size
    train_fraction=0.9
    val_fraction=0.099
    seed=7
    len_train_loader=int(len(dataset)*train_fraction/(multi_pass*batch_size))

    # Optimizer and scheduler
    lr = args.lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-9, weight_decay=0.01)
    num_steps_one_cycle = 120
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
                            model_type= 'minkowski' if use_sparse_cnn else 'transformer',
                            warmup_scheduler=warmup_scheduler,
                            loss_func=loss_fn,
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
                            save_model_path=f"{args.save_path}models/trackfit_model_{'sparse_cnn_' if use_sparse_cnn else ''}{j}.torch")
    
    if multi_GPU:
        torch.distributed.destroy_process_group()
    
    return model, training_dict    
    
    
    


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
    
    ## Get whether we will be using the sparse_cnn model or the transformer
    use_sparse_cnn=args.sparse_cnn
    



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
            if use_sparse_cnn:
                print(f"Using sparse_cnn model...")

            # generate dataset
            dataset=PGunEvent(root=args.dataset_folder,
                            shuffle=True,
                            multi_pass=multi_pass,
                            files_suffix='npz',
                            scaler_file=args.scaler_file,
                            use_true_tag=True,
                            scale_coordinates=(not use_sparse_cnn),
                            targets=args.targets,
                            inputs=args.inputs,
                            masking_scheme=args.ms)
            
            if args.targets is not None or args.weights!=loss_fn.weights:
                print(f"Selected targets are: "+"".join([f"{dataset.targets_names[k]} ({args.weights[dataset.targets[k]]:.1e})  " for k in range(len(dataset.targets_names))]))

            t0=time.perf_counter()
            

            if multi_GPU:
                # from sfgnets.track_fitting_net import main_worker
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

            # torch.save(model.state_dict(), f"/scratch4/maubin/models/hittag_model_{j}.torch")
            torch.save(training_dict,f"{args.save_path}results/trackfit_training_dict_{'sparse_cnn_' if use_sparse_cnn else ''}{j}.torch")
        
        if args.test or args.test_only:
            print("Testing the model...")
            testdataset_folder=args.dataset_folder[:-6]+"test/" ## we are assuming that the dataset_folder is of type "*_train/" whereas the testdataset folder will be "*_test/"
            
            testdataset=PGunEvent(root=testdataset_folder,
                        shuffle=False,
                        multi_pass=multi_pass,
                        files_suffix='npz',
                        scaler_file=args.scaler_file,
                        use_true_tag=True,
                        scale_coordinates=(not use_sparse_cnn),
                        targets=args.targets,
                        inputs=args.inputs,
                        masking_scheme=args.ms)
            
            if use_sparse_cnn:
                model=create_sparse_cnn_model(y_out_channels=sum(testdataset.targets_lengths)+sum(testdataset.targets_n_classes), x_in_channels=len(args.inputs) if args.inputs is not None else 2)
            else:
                model=create_transformer_model(y_out_channels=sum(testdataset.targets_lengths)+sum(testdataset.targets_n_classes), x_in_channels=len(args.inputs) if args.inputs is not None else 2)
            
            model.load_state_dict(torch.load(f"{args.save_path}models/trackfit_model_{'sparse_cnn_' if use_sparse_cnn else ''}{j}.torch"))
            
            full_loader=full_dataset(testdataset,
                                    collate=collate_minkowski if use_sparse_cnn else collate_transformer,
                                    batch_size=args.batch_size,)
            
            all_results=test_full(loader=full_loader,
                                                    model=model,
                                                    model_type="minkowski" if use_sparse_cnn else "transformer",
                                                    progress_bar=True,
                                                    device=device,
                                                    do_we_consider_aux=True,
                                                    do_we_consider_coord=True,
                                                    do_we_consider_feat=True,
                                                    # max_batches=10,
                                                    )
            perf_fn=SumPerf.from_SumLoss(loss_fn)

            if args.targets is not None:
                if args.mom_loss:
                    loss_fn.losses[1].loss_func=MomentumLoss()
                perf_fn=perf_fn[args.targets]
                # perf_fn.rebuild_partial_losses()

            all_results=measure_performances(all_results,
                                            testdataset,
                                            perf_fn,
                                            device, 
                                            model_type="minkowski" if use_sparse_cnn else "transformer",
                                            do_we_consider_aux=True,
                                            do_we_consider_coord=True,
                                            do_we_consider_feat=True,
                                            mom_spherical_coord=args.momsph)

            with open(f'{args.save_path}results/track_fitting_model_{"sparse_cnn_" if use_sparse_cnn else ""}{j}_pred.pkl', 'wb') as file:
                    pk.dump(all_results,file)
                    
            plots((all_results,testdataset),
                        plots_chosen=["pred_X","euclidian_distance","euclidian_distance_by_pdg","perf_charge","perf_distance", "euclidian_distance_by_primary", "perf_traj_length", "perf_kin_ener"],
                        savefig_path=f'{args.save_path}plots/track_fitting_model_{"sparse_cnn_" if use_sparse_cnn else ""}{j}.png',
                        model_name=str(j),
                        show=False,
                        )
            
            if args.targets is None or 1 in args.targets:
                plots((all_results,testdataset),
                        plots_chosen=["pred_X","euclidian_distance","euclidian_distance_by_pdg","perf_charge","perf_distance", "euclidian_distance_by_primary", "perf_traj_length", "perf_kin_ener"],
                        savefig_path=f'{args.save_path}plots/track_fitting_model_{"sparse_cnn_" if use_sparse_cnn else ""}{j}.png',
                        model_name=str(j),
                        mode='mom',
                        show=False,
                        )
                
                plots((all_results,testdataset),
                        plots_chosen=["pred_X","euclidian_distance","euclidian_distance_by_pdg","perf_charge","perf_distance", "euclidian_distance_by_primary", "perf_traj_length", "perf_kin_ener"],
                        savefig_path=f'{args.save_path}plots/track_fitting_model_{"sparse_cnn_" if use_sparse_cnn else ""}{j}.png',
                        model_name=str(j),
                        mode='mom_d',
                        show=False,
                        )
                
                plots((all_results,testdataset),
                        plots_chosen=["pred_X","euclidian_distance","euclidian_distance_by_pdg","perf_charge","perf_distance", "euclidian_distance_by_primary", "perf_traj_length", "perf_kin_ener"],
                        savefig_path=f'{args.save_path}plots/track_fitting_model_{"sparse_cnn_" if use_sparse_cnn else ""}{j}.png',
                        model_name=str(j),
                        mode='mom_n',
                        show=False,
                        )

    main()
