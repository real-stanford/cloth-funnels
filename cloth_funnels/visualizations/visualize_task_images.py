import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

#%%
# os.chdir("/local/crv/acanberk/folding-unfolding/src/")
hdf5_path = '/local/crv/acanberk/folding-unfolding/src/shirt-perturbation.hdf5'

def visualize_grid(images, n):
    next_divisible= (images.shape[0] // n + 1)
    # print(next_divisible_by_4)
    fig, axs = plt.subplots(n, next_divisible, \
        figsize=(next_divisible * 4, n * 4))
    for i in range(len(images)):
        # axs[0, i].imshow(unperturbed_images[i])
        axs[i%4, i//4].imshow(images[i])
    for ax in axs.flatten():
        ax.set_axis_off()
    fig.tight_layout()
    plt.show()
    fig.clear()

with h5py.File(hdf5_path, 'r') as dataset:
    for i, key in enumerate(dataset.keys()):
        group = dataset[key]
        perturbed_images = np.array(group['perturbed_images'])
        unperturbed_images = np.array(group['unperturbed_images'])

        print("Shape of perturbed images:", perturbed_images.shape)
        print("Shape of unperturbed images:", unperturbed_images.shape)

        if perturbed_images.shape[0] == 0:
            continue
     
        visualize_grid(perturbed_images, 4)
        visualize_grid(unperturbed_images, 4)


#%%

        # print(perturbed_images.shape[0])
        # next_divisible_by_4 = (perturbed_images.shape[0] // 4 + 1)
        # print(next_divisible_by_4)
        # fig, axs = plt.subplots(4, next_divisible_by_4, \
        #     figsize=(next_divisible_by_4 * 4, 16))
        # for i in range(len(perturbed_images)):
        #     # axs[0, i].imshow(unperturbed_images[i])
        #     axs[i%4, i//4].imshow(perturbed_images[i])
        # for ax in 
        # plt.show()
        # fig.clear()

        # fig, axs = plt.subplots(4, unperturbed_images.shape[0]//4, \
        #     figsize=(unperturbed_images.shape[0], 16))
        # for i in range(len(unperturbed_images)):
        #     axs[i//4, i%4].imshow(unperturbed_images[i])
        # plt.show()