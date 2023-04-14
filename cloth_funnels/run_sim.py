from cloth_funnels.utils.utils import (
    setup_envs,
    seed_all, 
    setup_network, 
    get_loader, 
    get_pretrain_loaders,
    get_dataset_size, 
    collect_stats, 
    get_img_from_fig,
    step_env, 
    visualize_value_pred)
import ray
from time import time, strftime
from copy import copy
import torch
from filelock import FileLock
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import wandb 
from nocs_model.network.orientation_deeplab import OrientationDeeplab
from cloth_funnels.learning.optimization import optimize_fb

from omegaconf import OmegaConf
import hydra
import yaml
import pathlib
import glob
import imageio.v3 as iio
import cv2
import shutil
import h5py
import sys
from notebooks.episode_visualizer import visualize_episode
from cloth_funnels.utils.utils import flatten_dict
from omegaconf import DictConfig, OmegaConf
import pathlib
import wandb

if __name__ == '__main__':

    # os.environ['WANDB_SILENT']='true'

    @hydra.main(version_base=None, config_path="conf", config_name="config")
    def main(args : DictConfig):

        assert args.name is not None or args.cont is not None, "Must name run or continue run"

        ray.init(local_mode=args.ray_local_mode)
        seed_all(args.seed)

        if args.name: # if name is not none or evaluating, create a new directory
            
            time_str = strftime("%m-%d-%H%M")       
            name = f"{time_str}-{args.name}"
            args.log = os.path.join(args.log_dir, name)

            pathlib.Path(args.log).mkdir(parents=True, exist_ok=True)

            wandb_run = wandb.init(
                project="cloth-funnels",
                config=flatten_dict(OmegaConf.to_container(args), sep="/"),  # type: ignore   
                tags=args.tags,
                notes=args.notes,
                dir=args.log_dir,
                name=name,
                mode = args.wandb
            )
            wandb_meta = {
                    'run_name': wandb_run.name,
                    'run_id': wandb_run.id
                }
            all_config = {
                'config': OmegaConf.to_container(args, resolve=True),
                'output_dir': os.getcwd(),
                'wandb': wandb_meta
                }
            yaml.dump(all_config, open(f'{args.log}/config.yaml', 'w'), default_flow_style=False)

        if args.cont:
            assert args.load is None, "Cannot load and continue at the same time"

            print(f"[RunSim] Continuing run {args.cont}")
            cont = args.cont

            all_config = yaml.load(open(f'{args.cont}/config.yaml', 'r'), Loader=yaml.FullLoader)
            args = OmegaConf.create(all_config['config'])

            args.cont = cont
            args.warmup = 0
        
            wandb_meta = all_config['wandb']
            wandb_run = wandb.init(
                project="cloth-funnels",
                config=flatten_dict(OmegaConf.to_container(args), sep="/"),  # type: ignore   
                tags=args.tags,
                notes=args.notes,
                dir=args.log_dir,
                name=wandb_meta['run_name'],
                id=wandb_meta['run_id'],
                resume=True,
                mode = args.wandb
            )

        # if args.load is not None:
        #     print(f"[RunSim] Loading from {args.load}")
        
        policy, optimizer, dataset_path = setup_network(args, gpu=args.network_gpu)

        criterion = torch.nn.functional.mse_loss
        nocs_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        envs, _ = setup_envs(dataset=dataset_path, **args)
        observations = ray.get([e.reset.remote() for e in envs])
        observations = [obs for obs, _ in observations]
        remaining_observations = []
        ready_envs = copy(envs)
 
        dataset_size = get_dataset_size(dataset_path) 
        init_dataset_size = dataset_size
        script_init_time = time()

        step_iter = 0

        while(True):
            with torch.no_grad():
                print("[RunSim] Stepping env")
                ready_envs, observations, remaining_observations =\
                    step_env(
                        all_envs=envs,
                        ready_envs=ready_envs,
                        ready_actions=policy.act(observations),
                        remaining_observations=remaining_observations,
                        deterministic=args.deterministic)
                   
            if dataset_size > args.warmup and args.warmup_only:
                sys.exit("[RunSim] Warmup ended, terminating...")

            if (optimizer is not None) and (dataset_size > args.warmup) and not args.eval:
                should_update = (dataset_size // args.points_per_update) > (policy.train_steps // args.batches_per_update)
                if should_update:
                    policy.train()

                    with FileLock(dataset_path + ".lock"):
                        start = time()
                        loaders = (get_loader(
                                hdf5_path=dataset_path,
                                filter_fn=lambda attrs:
                                    (attrs['action_primitive'] == action_primitive \
                                        and ('preaction_l2_distance' in attrs)), **args) for action_primitive in args.action_primitives)
                        loader = zip(*loaders)
                        print("Loaded dataset in {} seconds".format(time() - start))
                        optimize_fb(
                            policy=policy,
                            optimizer=optimizer,
                            loader=loader,
                            criterion=criterion,
                            nocs_criterion=nocs_criterion,
                            num_updates=args.batches_per_update,
                            dataset_size=dataset_size,
                            **args)
                        del loader

                    policy.decay_exploration(args.value_expl_halflife, args.action_expl_halflife, dataset_size, args.points_per_update, is_eval=args.eval)
                    policy.eval()
                   
                NUM_SAVES=5
                ckpt_paths = glob.glob(f"{args.log}/ckpt_*.pth")
                ckpt_paths.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
                for ckpt_path in ckpt_paths[:-NUM_SAVES]:
                    os.remove(ckpt_path)
            
                checkpoint_paths = [f'{args.log}/latest_ckpt.pth']
                checkpoint_paths.append(
                        f'{args.log}/ckpt_{policy.train_steps}.pth')

                for path in checkpoint_paths:
                    save_dict = {
                        'optimizer': optimizer.state_dict(),
                        'net': policy.state_dict()
                    }
                    torch.save(save_dict,  path)

            dataset_size = get_dataset_size(dataset_path)
            pph = 3600 * (dataset_size - init_dataset_size)/(time() - script_init_time)
            print("[RunSim] Points per hour:", pph)

            if dataset_size > args.max_steps:
                print("[RunSim] Experiment concluded")
                exit(0)
            
            if dataset_size > 32 and (step_iter % 8 == 0) and not (args.fold_finish):
                # print("Logging for step:", i)
                start = time()
                stats_dict = collect_stats(dataset_path, num_points=512)

                end = time()
                print(f"Collecting stats took {end-start} seconds")

                pph = 3600 * (dataset_size - init_dataset_size)/(time() - script_init_time)
                print('='*18 + f' {dataset_size} points ({pph} p/h) ' + '='*18)

                # writer.add_scalar("points_per_hours", pph, global_step=dataset_size)
                wandb.log({"points_per_hour": pph}, step=dataset_size)
                wandb.log({'action_expl_prob': float(policy.action_expl_prob)}, step=dataset_size)
                wandb.log({'value_expl_prob': float(policy.value_expl_prob)}, step=dataset_size)

                # try:
                    # with h5py.File(dataset_path, "r") as dataset:
                if step_iter % 16 == 0:
                    fig, axs, _, _ = visualize_episode(stats_dict['vis_key'], dataset_path, steps=args.episode_length, vis_index=(0, 1, 2, 3))
                    try:
                        wandb.log({"img_episode_vis": wandb.Image(fig)}, step=dataset_size)
                    except: 
                        print("[RunSim] Could not visualize episode, file unavailable")
                        pass
                del stats_dict['vis_key']
                # except:
                #     print("[RunSim] Could not visualize episode, file unavailable")
                #     pass
                   
                start = time()
                try:
                    for key, value in stats_dict.items():
                        if 'distribution' in key:
                            sequence = np.array(value, dtype=np.float32)
                            data = [[s] for s in sequence]
                            table = wandb.Table(data=data, columns=["scales"])
                            wandb.log({f'histogram_{key}': wandb.plot.histogram(table, "scales",
                                title=f"{key}")}, step=dataset_size)

                        elif 'img' in key:
                            value = (np.array(value).astype(np.uint8))[:3, :, :].transpose(1, 2, 0)
                            wandb.log({key: wandb.Image(value)}, step=dataset_size)
                        else:
                            wandb.log({key: float(value)}, step=dataset_size)
                    try:
                        videos = glob.glob(f'{args.log}/videos/*/*.mp4')
                        if len(videos):
                            select_vid = np.random.choice(videos)
                            #load video as numpy array
                            frames = []
                            for i, frame in enumerate(iio.imiter(select_vid)):
                                frames.append(cv2.resize(frame, (128, 128)))
                            frames = np.stack(frames, axis=0)
                            wandb.log({'video': wandb.Video(frames.transpose(0, 3, 1, 2), fps=24)}, step=dataset_size)
                            
                            if not args.dump_visualizations:
                                for video_dir in glob.glob(f'{args.log}/videos/*'):
                                    shutil.rmtree(video_dir, ignore_errors=True)

                        print(f"Logging took {time()-start} seconds")
                    except Exception as e:
                        print("[Video Could Not Be Uploaded]", e)
                        pass
                except Exception as e:
                    print("[RunSim] Could not log stats", e)
                    pass
            step_iter += 1


    main()