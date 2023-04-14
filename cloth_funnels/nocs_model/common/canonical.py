import numpy as np
import scipy.ndimage as sn

def nocs_to_direction(
        nocs_img: np.ndarray,
        sigma: float=1,
        eps: float=1e-7,
        ) -> np.ndarray:
    """
    nocs_img: H,W,C
    return: H,W,C,2(dh dw)
    """
    grads = np.moveaxis(np.stack([sn.gaussian_filter1d(
        nocs_img, sigma=sigma, axis=i, order=1)
        for i in [0,1]]),0,-1)
    inv_norms = 1/np.maximum(np.linalg.norm(grads,axis=-1),eps)
    directions = np.einsum('abcd,abc->abcd', grads, inv_norms)
    return directions

def direction_to_angle(
        direction_img: np.ndarray
        ) -> np.ndarray:
    """
    Counter-clockwise from down vector in image space.
    I.e. if cloth face right, NOCS Z angle should be pi/2 (90 deg)
    In range 0 to 2*pi

    direction_img: H,W,C,2
    return: H,W,C
    """
    angles = np.arctan2(direction_img[...,1], direction_img[...,0]) % (np.pi * 2)
    return angles

def angles_to_bin(
        angle_img: np.ndarray,
        period: float=2*np.pi,
        bins: int=32,
        dtype = np.uint8
        ) -> np.ndarray:
    """
    
    angle_img: H,W,C
    return: H,W,C dtype
    """
    norm_angles = (angle_img % period) / period
    bin_ids = np.minimum((norm_angles * bins).astype(dtype),bins-1)
    return bin_ids

def test():
    from skimage.io import imread
    from matplotlib import pyplot as plt
    nocs_img = imread('/home/cchi/dev/folding-unfolding/src/nocs_model/data/nocs.png')
    nocs_img = nocs_img[...,:3].astype(np.float32) / 255

    direction_img = nocs_to_direction(nocs_img)
    angle_img = direction_to_angle(direction_img)
    bins_img = angles_to_bin(angle_img)
