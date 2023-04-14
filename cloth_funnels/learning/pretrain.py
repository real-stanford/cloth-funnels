   #pretrain on filtered dataset
        # if args.pretrain_dataset_path is not None and optimizer is not None:

        #     pretrain_dataset_path = args.pretrain_dataset_path
        #     checkpoint_paths = [f'{args.log}/pretrain/latest_ckpt.pth']

        #     #create directory log + /pretrain/
        #     if not os.path.exists(args.log + "/pretrain/"):
        #         os.makedirs(args.log + "/pretrain/")

        #     #make directory log + /vis/
        #     if not os.path.exists(args.log + "/vis/"):
        #         os.makedirs(args.log + "/vis/")
        #     else:
        #         #delete vis and recreate it
        #         shutil.rmtree(args.log + "/vis/")
        #         os.makedirs(args.log + "/vis/")

        #     def pretrain_filter(group):
        #         # return group.attrs['grasp_success']
        #         return True

        #     with FileLock(pretrain_dataset_path + ".lock"):
        #         print("Loading dataset")
        #         start = time()
        #         pretrain_loaders = [get_loader(
        #                             supervised_training=True,
        #                             hdf5_path=pretrain_dataset_path,
        #                             replay_buffer_size=500,
        #                             filter_fn=lambda attrs:
        #                                 (attrs['action_primitive'] == action_primitive), **vars(args)) for action_primitive in args.action_primitives]
        #         pretrain_loader = zip(*pretrain_loaders)
        #         print("Loaded dataset in {} seconds".format(time() - start))
            
        #     for epoch in range(args.pretrain_epochs):
        #         optimize_fb(
        #                 policy=policy,
        #                 target_policy=target_policy,
        #                 # orientation_net=policy.orientation_network,
        #                 optimizer=optimizer,
        #                 loader=pretrain_loader,
        #                 criterion=criterion,
        #                 nocs_criterion=nocs_criterion,
        #                 writer=writer,
        #                 num_updates = min([len(loader) for loader in pretrain_loaders]),
        #                 **vars(args))

                
        #         for path in checkpoint_paths:
        #             save_dict = {
        #                 'optimizer': optimizer.state_dict(),
        #                 'net': policy.state_dict()
        #             }
        #             torch.save(save_dict,  path)

        #     exit(1)