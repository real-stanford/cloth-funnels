#%%
import os 
# os.chdir('/local/crv/acanberk/cloth-funnels/cloth_funnels')
import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms.functional as TF
from cloth_funnels.keypoint_detector.networks.keypoint_deeplab import KeypointDeeplab
from cloth_funnels.keypoint_detector.common.torch_util import to_numpy
from scipy.ndimage import rotate

#%%

# class KeypointModel():
#     def __init__(self, model_path):
#         self.model_path = model_path
#         dlc_proc = Processor()
#         self.dlc_live = DLCLive(model_path, processor=dlc_proc)

#     def get_keypoints(self, image, mask=None):
#         """
#         Takes in 3x720x720 image, returns 10x3 keypoints
#         """
#         assert image.shape[0] == 3
#         image = image.transpose(1, 2, 0)

#         self.dlc_live.init_inference(image)
#         coordinates = self.dlc_live.get_pose(image)

#         #if mask is not none, ensure that every point is on the cloth
#         if mask is not None:
#             mask_coords = np.array(np.where(mask))
#             for i, coord in enumerate(coordinates):
#                 xy = np.array((coord[1], coord[0]))
#                 if not mask[int(coord[1]), int(coord[0])]:
#                     nearest_coord_idx = np.argmin(np.linalg.norm((xy.reshape(2, 1) - mask_coords), axis=0))
#                     nearest_coord = mask_coords[:, nearest_coord_idx]
#                     # print("coord", coord)
#                     coordinates[i, 0] = nearest_coord[1]
#                     coordinates[i, 1] = nearest_coord[0]
#         return coordinates


class KeypointDetector():
    def __init__(self, model_path, input_size=128):

        self.model_path = model_path
        self.network = KeypointDeeplab.load_from_checkpoint(model_path).float()
        self.network.eval()

    def get_keypoints(self, image, mask=None, model_dim=128):
        """
        Takes in NxNx3 image, returns Nx3 keypoints
        """
        assert image.shape[-1] == 3

        original_dims = image.shape[-2:]

        image = cv2.resize(image, (model_dim, model_dim))

        image = (image - np.array([0.5, 0.5, 0.5]))/np.array([0.5, 0.5, 0.5])
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, 0)
        image = torch.tensor(image).float()

        with torch.no_grad():
            out = self.network.forward(image)

        scoremap = to_numpy(out['scoremap'][0])
        scoremap_max_idx = np.argmax(scoremap.reshape(scoremap.shape[0], -1), axis=-1)
        scoremap_max_np = np.stack(np.unravel_index(scoremap_max_idx, shape=scoremap.shape[-2:])).T[:,::-1].astype(np.float64)
        scoremap_max_np *= (original_dims[0]/model_dim)

        return scoremap_max_np.astype(np.int16)

   
if __name__ == "__main__":

    model_path = '/local/crv/acanberk/folding-unfolding/src/outputs/2022-08-26/16-32-41/checkpoints/epoch=292-val_keypoint_dist=2.9334.ckpt'
    model = KeypointDetector(model_path)

    import skvideo.io

    video_path = '/local/crv/acanberk/folding-unfolding/src/shirt_data/buffer0_video.mp4'
    
    def draw_keypoint(img, pred_keypoint):
        cv2.drawMarker(img, pred_keypoint, 
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=20, thickness=4)
    #read video from video apth
    num_examples = 10
    video = skvideo.io.vread(video_path)
    frame_indices = np.random.choice(np.arange(video.shape[0]), size=num_examples, replace=False)
    frames = video[frame_indices]

    fig, axs = plt.subplots(1, num_examples, figsize=(num_examples*2, 2))

    for i, frame in enumerate(frames):

        frame = frame[150:-150, 150:-150]
        obs = frame.copy()
        # print(frame.min(), frame.max())
        # print(frame.shape)
        kps = model.get_keypoints(frame.copy()/255)

        obs = np.ascontiguousarray(frame)

        for j, kp in enumerate(kps):
            draw_keypoint(obs, kp)
        
        axs[i].imshow(obs)
        axs[i].axis('off')
    
    plt.show()


        


    # with h5py.File('/local/crv/acanberk/folding-unfolding/src/experiments/03-dc2c-fr-unfn/latest_ckpt_eval_4/replay_buffer.hdf5') as dataset:
    #     keys = list(dataset.keys())
    #     np.random.shuffle(keys)
    #     num_samples = 5
    #     fig, axs = plt.subplots(1, num_samples, figsize=(num_samples*2, 2))

    #     for i, key in enumerate(keys):
    #         group = dataset[key]
            
    #         obs = group['next_observations'][:3].transpose(1, 2, 0)

    #         obs = rotate(obs, np.random.random() * 180)
     
    #         kps = model.get_keypoints(obs)
    #         print(kps)

    #         def draw_keypoint(img, pred_keypoint):
    #             cv2.drawMarker(img, pred_keypoint, 
    #                     color=(255,0,0), markerType=cv2.MARKER_CROSS,
    #                     markerSize=20, thickness=4)

    #         obs = np.ascontiguousarray((obs * 255).astype(np.uint8))
    #         for k in kps:
    #             draw_keypoint(obs, k)

    #         axs[i].imshow(obs)

    #         if i == num_samples - 1:
    #             break

    #     fig.tight_layout()
    #     plt.tight_layout()
    #     for ax in axs.flat:
    #         ax.set_axis_off()
    #     plt.show()


    # model_path = '/local/crv/acanberk/folding-unfolding/src/outputs/2022-08-26/16-32-41/checkpoints/epoch=292-val_keypoint_dist=2.9334.ckpt'
    # model = KeypointDetector(model_path)
    # with h5py.File('/local/crv/acanberk/folding-unfolding/src/experiments/03-dc2c-fr-unfn/latest_ckpt_eval_4/replay_buffer.hdf5') as dataset:
    #     keys = list(dataset.keys())
    #     np.random.shuffle(keys)
    #     num_samples = 5
    #     fig, axs = plt.subplots(1, num_samples, figsize=(num_samples*2, 2))

    #     for i, key in enumerate(keys):
    #         group = dataset[key]
            
    #         obs = group['next_observations'][:3].transpose(1, 2, 0)

    #         obs = rotate(obs, np.random.random() * 180)
     
    #         kps = model.get_keypoints(obs)
    #         print(kps)

    #         def draw_keypoint(img, pred_keypoint):
    #             cv2.drawMarker(img, pred_keypoint, 
    #                     color=(255,0,0), markerType=cv2.MARKER_CROSS,
    #                     markerSize=20, thickness=4)

    #         obs = np.ascontiguousarray((obs * 255).astype(np.uint8))
    #         for k in kps:
    #             draw_keypoint(obs, k)

    #         axs[i].imshow(obs)

    #         if i == num_samples - 1:
    #             break

    #     fig.tight_layout()
    #     plt.tight_layout()
    #     for ax in axs.flat:
    #         ax.set_axis_off()
    #     plt.show()



# %%
