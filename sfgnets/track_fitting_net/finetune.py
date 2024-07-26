from .train import *

import argparse
    
parser = argparse.ArgumentParser(
                    prog='TrackFittingFinetuning',
                    description='Fine tunes a model for Track Fitting in SFG',)

parser.add_argument('j', metavar='j', type=int, help='#j div of the model')
# parser.add_argument('dataset_folder',metavar='Dataset_Folder', type=str, help="Folder in which are stored the event_#.npz files for training")
# parser.add_argument('scaler_file',metavar='Scaler_File', type=str, help="File storing the dataset features scalers")
# parser.add_argument('save_path',metavar='Save_Path', type=str, help="Path to save results and models")
# parser.add_argument('-B', '--sparse_cnn', action='store_true', help='use the sparse_cnn model (MinkUNet), otherwise use the transformer')
# parser.add_argument('-m', '--multi_GPU', action='store_true', help='runs the script on multi GPU')
# parser.add_argument('-b', '--benchmarking', action='store_true', help='prints the duration of the different parts of the code')
# parser.add_argument('-s', '--sub_tqdm', action='store_true', help='displays the progress bars of the train and test loops for each epoch')
parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
parser.add_argument('-n','--num_epochs', type=int, default=3, help='number of epochs for the training')
parser.add_argument('--test_only', action='store_true', help='runs only the test (measure performances, plots, ...)')
parser.add_argument('-T', '--test', action='store_true', help='runs test after training (measure performances, plots, ...)')
# parser.add_argument('--multi_pass', type=int, default=1, help='how many times the whole dataset is gone through in an epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='maximum learning rate for training (defines the scale of the learning rate)')
# parser.add_argument('--stop_after_epochs', type=int, default=5, help='maximum number of epochs without improvement before stopping the training (early termination)')
parser.add_argument('-w','--weights', type=float, nargs=len(loss_fn), default=loss_fn.weights, help='weights for the loss functions')
parser.add_argument('-t','--targets', type=int, nargs="*", default=None, help='the target indices to include')
parser.add_argument('-i','--inputs', type=int, nargs="*", default=None, help='the input indices to include')
parser.add_argument('--ms', type=str, default='distance', metavar='Masking Scheme', help='Which mask to use')
parser.add_argument('-S','--steps', type=int, nargs="*", default=[1,2,3], help='which step to execute')
parser.add_argument('--mom_loss', action='store_true', help='use the momentum loss instead of the simple MSE')  
parser.add_argument('--momdir_loss', action='store_true', help='use the momentum-direction loss instead of the simple MSE')  
parser.add_argument("-G", "--gpu", type=int, default=1, help="GPU ID (cuda) to be used")
args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



j=args.j

use_sparse_cnn=False

if args.mom_loss:
    loss_fn.losses[1].loss_func=MomentumLoss()
elif args.momdir_loss:
    loss_fn=loss_fn_momdir_loss

if not args.test_only:
    
    dataset=PGunEvent(root="/scratch4/maubin/data/pgun_train",
                            shuffle=True,
                            multi_pass=1,
                            files_suffix='npz',
                            # scaler_file="/scratch4/maubin/data/scaler_trackfit.p",
                            # scaler_file="/scratch4/maubin/data/scaler_trackfit_saul.p",
                            # scaler_file="/scratch4/maubin/data/scaler_trackfit2.p",
                            # scaler_file="/scratch4/maubin/data/scaler_trackfit_intermediate.p",
                            # scaler_file=None,
                            use_true_tag=True,
                            scale_coordinates=True,
                            targets=args.targets,
                            inputs=args.inputs,
                            masking_scheme=args.ms)
    
    if args.targets is not None:
        ## Select only the targets for the loss function
        loss_fn=loss_fn[dataset.targets]
        
        ## Replace the weights of the loss function
        loss_fn.weights=[args.weights[k] for k in dataset.targets]




    model=create_transformer_model(y_out_channels=sum(dataset.targets_lengths)+sum(dataset.targets_n_classes), x_in_channels=len(args.inputs) if args.inputs is not None else 2)
    model.load_state_dict(torch.load(f"/scratch4/maubin/models/trackfit_model_{j}.torch"),strict=False)


    if 1 in args.steps:
        ############## FINETUNING THE DECODER ##############

        print("Starting with the finetuning of the decoder...")

        dataset=PGunEvent(root="/scratch4/maubin/data/pgun_train",
                                shuffle=True,
                                multi_pass=1,
                                files_suffix='npz',
                                # scaler_file="/scratch4/maubin/data/scaler_trackfit.p",
                                # scaler_file="/scratch4/maubin/data/scaler_trackfit_saul.p",
                                # scaler_file="/scratch4/maubin/data/scaler_trackfit2.p",
                                scaler_file="/scratch4/maubin/data/scaler_trackfit_intermediate.p",
                                # scaler_file=None,
                                use_true_tag=True,
                                scale_coordinates=True,
                                targets=args.targets,
                                inputs=args.inputs,
                                masking_scheme=args.ms)

        # Freezing the whole model
        for param in model.parameters():
            param.requires_grad = False

        # Unfreezing the decoder
        for param in model.decoder.parameters():
            param.requires_grad = True


        # Loaders parameters
        batch_size=args.batch_size
        train_fraction=0.9
        val_fraction=0.099
        seed=7
        multi_pass=1
        len_train_loader=int(len(dataset)*train_fraction/(multi_pass*batch_size))

        # Optimizer and scheduler
        lr = args.lr*10
        optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-9, weight_decay=0.01)
        num_steps_one_cycle = 2
        num_warmup_steps = 1
        cosine_annealing_steps = len_train_loader * num_steps_one_cycle

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(cosine_annealing_steps), T_mult=1, eta_min=lr*(1e-2))
        warmup_scheduler = WarmUpScheduler(optimizer, lr_scheduler,
                                        len_loader=1,
                                        warmup_steps=int(len_train_loader * num_warmup_steps),
                                        warmup_start_lr=lr/100,
                                        warmup_mode='linear',
                                        )

        num_epochs=args.num_epochs
        stop_after_epochs=5

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
                                benchmarking=False,
                                sub_progress_bars=True,
                                multi_GPU=False,
                                world_size=1,
                                save_model_path=f"/scratch4/maubin/models/trackfit_model_{'sparse_cnn_' if use_sparse_cnn else ''}{j}_finetuned_intermediate.torch",
                                num_workers=24,
                                # notebook_tqdm=True,
                                )

    if 2 in args.steps:
        ############## FINETUNING THE ENCODER ##############

        print("Pursuing with the finetuning of the encoder...")

        dataset=PGunEvent(root="/scratch4/maubin/data/pgun_train",
                                shuffle=True,
                                multi_pass=1,
                                files_suffix='npz',
                                # scaler_file="/scratch4/maubin/data/scaler_trackfit.p",
                                # scaler_file="/scratch4/maubin/data/scaler_trackfit_saul.p",
                                scaler_file="/scratch4/maubin/data/scaler_trackfit2.p",
                                # scaler_file="/scratch4/maubin/data/scaler_trackfit_intermediate.p",
                                # scaler_file=None,
                                use_true_tag=True,
                                scale_coordinates=True,
                                targets=args.targets,
                                inputs=args.inputs,
                                masking_scheme=args.ms)

        # Freezing the whole model
        for param in model.parameters():
            param.requires_grad = False

        # Unfreezing the decoder
        for param in model.transformer_encoder.parameters():
            param.requires_grad = True


        # Loaders parameters
        batch_size=args.batch_size
        train_fraction=0.9
        val_fraction=0.099
        seed=7
        multi_pass=1
        len_train_loader=int(len(dataset)*train_fraction/(multi_pass*batch_size))

        # Optimizer and scheduler
        lr = args.lr*10
        optimizer = torch.optim.AdamW(model.transformer_encoder.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-9, weight_decay=0.01)
        num_steps_one_cycle = 2
        num_warmup_steps = 1
        cosine_annealing_steps = len_train_loader * num_steps_one_cycle

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=int(cosine_annealing_steps), T_mult=1, eta_min=lr*(1e-2))
        warmup_scheduler = WarmUpScheduler(optimizer, lr_scheduler,
                                        len_loader=1,
                                        warmup_steps=int(len_train_loader * num_warmup_steps),
                                        warmup_start_lr=lr/100,
                                        warmup_mode='linear',
                                        )

        num_epochs=args.num_epochs
        stop_after_epochs=5

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
                                benchmarking=False,
                                sub_progress_bars=True,
                                multi_GPU=False,
                                world_size=1,
                                save_model_path=f"/scratch4/maubin/models/trackfit_model_{'sparse_cnn_' if use_sparse_cnn else ''}{j}_finetuned_intermediate.torch",
                                num_workers=24,
                                # notebook_tqdm=True,
                                )
        
    if 3 in args.steps:

        ############## FINETUNING THE FULL MODEL ##############

        print("Now finetuning the whole model...")

        dataset=PGunEvent(root="/scratch4/maubin/data/pgun_train",
                                shuffle=True,
                                multi_pass=1,
                                files_suffix='npz',
                                # scaler_file="/scratch4/maubin/data/scaler_trackfit.p",
                                # scaler_file="/scratch4/maubin/data/scaler_trackfit_saul.p",
                                scaler_file="/scratch4/maubin/data/scaler_trackfit2.p",
                                # scaler_file="/scratch4/maubin/data/scaler_trackfit_intermediate.p",
                                # scaler_file=None,
                                use_true_tag=True,
                                scale_coordinates=True,
                                targets=args.targets,
                                inputs=args.inputs,
                                masking_scheme=args.ms)

        # Unfreezing the whole model
        for param in model.parameters():
            param.requires_grad = True


        # Loaders parameters
        batch_size=args.batch_size
        train_fraction=0.9
        val_fraction=0.099
        seed=7
        multi_pass=1
        len_train_loader=int(len(dataset)*train_fraction/(multi_pass*batch_size))

        # Optimizer and scheduler
        lr = args.lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-9, weight_decay=0.01)
        num_steps_one_cycle = 30
        num_warmup_steps = 2
        cosine_annealing_steps = len_train_loader * num_steps_one_cycle

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cosine_annealing_steps, T_mult=1, eta_min=lr*(1e-2))
        warmup_scheduler = WarmUpScheduler(optimizer, lr_scheduler,
                                        len_loader=1,
                                        warmup_steps=len_train_loader * num_warmup_steps,
                                        warmup_start_lr=lr/100,
                                        warmup_mode='linear')

        num_epochs=args.num_epochs
        stop_after_epochs=25

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
                                benchmarking=False,
                                sub_progress_bars=True,
                                multi_GPU=False,
                                world_size=1,
                                save_model_path=f"/scratch4/maubin/models/trackfit_model_{'sparse_cnn_' if use_sparse_cnn else ''}{j}_finetuned.torch",
                                num_workers=24,
                                # notebook_tqdm=True,
                                )
        
    
    torch.save(training_dict,f"/scratch4/maubin/results/trackfit_training_dict_{'sparse_cnn_' if use_sparse_cnn else ''}{j}_finetuned.torch")


############## TESTING THE FULL MODEL ##############

if args.test or args.test_only:
    print("Testing the model...")
    testdataset_folder="/scratch4/maubin/data/pgun_test"
    
    testdataset=PGunEvent(root=testdataset_folder,
                shuffle=False,
                multi_pass=1,
                files_suffix='npz',
                scaler_file="/scratch4/maubin/data/scaler_trackfit2.p",
                use_true_tag=True,
                scale_coordinates=(not use_sparse_cnn),
                targets=args.targets,
                inputs=args.inputs,
                masking_scheme=args.ms)
    
    if use_sparse_cnn:
        model=create_sparse_cnn_model(y_out_channels=sum(testdataset.targets_lengths)+sum(testdataset.targets_n_classes), x_in_channels=len(args.inputs) if args.inputs is not None else 2)
    else:
        model=create_transformer_model(y_out_channels=sum(testdataset.targets_lengths)+sum(testdataset.targets_n_classes), x_in_channels=len(args.inputs) if args.inputs is not None else 2)
    
    model.load_state_dict(torch.load(f"/scratch4/maubin/models/trackfit_model_{'sparse_cnn_' if use_sparse_cnn else ''}{j}_finetuned.torch"))
    
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
        perf_fn.rebuild_partial_losses()

    all_results=measure_performances(all_results,
                                    testdataset,
                                    perf_fn,
                                    device, 
                                    model_type="minkowski" if use_sparse_cnn else "transformer",
                                    do_we_consider_aux=True,
                                    do_we_consider_coord=True,
                                    do_we_consider_feat=True,)

    with open(f'/scratch4/maubin/results/track_fitting_model_{"sparse_cnn_" if use_sparse_cnn else ""}{j}_finetuned_pred.pkl', 'wb') as file:
            pk.dump(all_results,file)
            
    plots.plots((all_results,testdataset),
                plots_chosen=["pred_X","euclidian_distance","euclidian_distance_by_pdg","perf_charge","perf_distance", "euclidian_distance_by_primary", "perf_traj_length", "perf_kin_ener"],
                savefig_path=f'/scratch4/maubin/plots/track_fitting_model_{"sparse_cnn_" if use_sparse_cnn else ""}{j}_finetuned.png',
                model_name=str(j),
                show=False,
                )
    
    if args.targets is None or 1 in args.targets:
        plots.plots((all_results,testdataset),
                plots_chosen=["pred_X","euclidian_distance","euclidian_distance_by_pdg","perf_charge","perf_distance", "euclidian_distance_by_primary", "perf_traj_length", "perf_kin_ener"],
                savefig_path=f'/scratch4/maubin/plots/track_fitting_model_{"sparse_cnn_" if use_sparse_cnn else ""}{j}_finetuned.png',
                model_name=str(j),
                mode='mom',
                show=False,
                )
        
        plots.plots((all_results,testdataset),
                plots_chosen=["pred_X","euclidian_distance","euclidian_distance_by_pdg","perf_charge","perf_distance", "euclidian_distance_by_primary", "perf_traj_length", "perf_kin_ener"],
                savefig_path=f'/scratch4/maubin/plots/track_fitting_model_{"sparse_cnn_" if use_sparse_cnn else ""}{j}_finetuned.png',
                model_name=str(j),
                mode='mom_d',
                show=False,
                )
        
        plots.plots((all_results,testdataset),
                plots_chosen=["pred_X","euclidian_distance","euclidian_distance_by_pdg","perf_charge","perf_distance", "euclidian_distance_by_primary", "perf_traj_length", "perf_kin_ener"],
                savefig_path=f'/scratch4/maubin/plots/track_fitting_model_{"sparse_cnn_" if use_sparse_cnn else ""}{j}_finetuned.png',
                model_name=str(j),
                mode='mom_n',
                show=False,
                )