# %%
import h5py
from filelock import FileLock


replay_buffer = "/local/crv/acanberk/folding-unfolding/src/flingbot_eval_2/replay_buffer.hdf5"

with FileLock(replay_buffer + ".lock"):
    with h5py.File(replay_buffer, 'a') as f:
        print(list(f.keys()))

# %%



# %%

if __name__ == "__main__":
    import os
    from learning.nocs_unet_inference import nocs_pred, flip_nocs, correspondence_loss_from_rgbd, pickle_to_rgbd_nocs
    import numpy as np
    import matplotlib.pyplot as plt
    import torchvision
    import torch
    import cv2
    import pickle as pkl

    replay_buffer_path = "/local/crv/acanberk/folding-unfolding/src/correspondence_images_"
    target_nocs_path = "/local/crv/acanberk/folding-unfolding/src/learning/target_nocs_1.pkl"
    
    # #load pickle from target_nocs_path
    # rgbd, target_nocs_a = pickle_to_rgbd_nocs(target_nocs_path)
    # #rotate target_nocs_a image clockwise 180 degrees
    # target_nocs_a = cv2.rotate(target_nocs_a, cv2.ROTATE_180)
    # data = [rgbd, target_nocs_a]
    # #save data to target_nocs_path
    # pkl.dump(data, open(target_nocs_path, 'wb'))

    # #load pickle from target_nocs_path
    # rgbd, target_nocs_a = pickle_to_rgbd_nocs(target_nocs_path)
    # #visulize target_nocs_a
    # plt.imshow(target_nocs_a)
    # plt.show()

    

    
    num_figs = 10

    _, target_nocs_a = pickle_to_rgbd_nocs(target_nocs_path)
    target_nocs_b = flip_nocs(target_nocs_a)


    for file in os.listdir(replay_buffer_path):
        #load numpy array from .npy and plot
        print(file)
        env_path = os.path.join(replay_buffer_path, file)
        steps = os.listdir(env_path)
        steps.sort()
        print(steps)

        for step_n in range(5):

            step_path = os.path.join(env_path, steps[step_n])
            
            #create a 5,2 plot
            fig, axs = plt.subplots(num_figs, 6, figsize=(12, num_figs*2.5))

            rgbd_instance, gt_nocs_instance = pickle_to_rgbd_nocs(step_path)
            floor_depth = np.max(rgbd_instance[:,:,3])
            print("Floor depth ", floor_depth)

            for i in range(num_figs):
                ###

                degrees = 360
                translate = (0.25, 0.25)
                scale = (0.5, 1.5)
                shear = 0.0

                rgbd, gt_nocs = rgbd_instance.copy(), gt_nocs_instance.copy()

                transformation = torchvision.transforms.RandomAffine(degrees=degrees, \
                    translate=translate, scale=scale, shear=shear, \
                        interpolation=torchvision.transforms.InterpolationMode.NEAREST)
                
                rgbd = torch.tensor(rgbd.transpose(2, 0, 1))
                rgbd = transformation(rgbd)
                rgbd = rgbd.numpy().transpose(1, 2, 0)


                d = rgbd[:,:,3]
                # plt.imshow(d)
                # plt.show()
                zero_depth_indices = np.where(d < 0.2)
                d[zero_depth_indices] = floor_depth
                # print(np.min(d))
                # plt.imshow(d)
                # plt.show()

                rgbd[:, :, 3] = d
                input_nocs = nocs_pred(rgbd)
                correspondence_loss = correspondence_loss_from_rgbd(rgbd)
                #at axs[i] plot input and target
                axs[i, 0].imshow(rgbd[:, :, :3].astype(np.int16))
                axs[i, 0].set_axis_off()
                axs[i, 0].set_title("in rgbd")

                axs[i, 1].imshow(rgbd[:, :, 3])
                axs[i, 1].set_axis_off()
                axs[i, 1].set_title("in depth")

                axs[i, 2].imshow(input_nocs)
                axs[i, 2].set_axis_off()
                axs[i, 2].set_title("L={:.1f}".format(correspondence_loss))

                axs[i, 3].imshow(gt_nocs)
                axs[i, 3].set_title("gt_nocs")
                axs[i, 3].set_axis_off()

                axs[i, 4].imshow(target_nocs_a)
                axs[i, 4].set_title("target_nocs_a")
                axs[i, 4].set_axis_off()

                axs[i, 5].set_title("target_nocs_b")
                axs[i, 5].imshow(target_nocs_b)
                axs[i, 5].set_axis_off()

            plt.show()

# %%
