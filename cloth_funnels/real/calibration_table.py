
import time
import numpy as np
import cv2
import pickle
from scipy import optimize

from cair_robot.cameras.kinect_client import KinectClient

cam = KinectClient('128.59.23.32', '8080')
cam_intr = cam.get_intr()
print(cam_intr)

def capture_images(num_calib_pts=12):
    all_img = []
    observed_pts = []
    observed_pix = []
    measured_pts = []
    while(len(measured_pts) < num_calib_pts):
        input('Press Enter to take image')
        while True:
            color_im, depth_im = cam.get_camera_data(n=5)
            chckr_size = (3, 3)
            refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            #bgr_im = cv2.cvtColor(color_im, cv2.COLOR_RGB2BGR)
            #gray_im = cv2.cvtColor(bgr_im, cv2.COLOR_RGB2GRAY)
            gray_im = color_im[:, :, 0] - np.mean(color_im[:,:,[1,2]], axis=-1)
            gray_im -= np.min(gray_im)
            gray_im /= np.max(gray_im)
            gray_im = (255 - gray_im * 255.0).astype(np.uint8)
            chckr_found, crnrs = cv2.findChessboardCorners(gray_im, chckr_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH)
            if chckr_found:
                crnrs_refined = cv2.cornerSubPix(gray_im, crnrs, (4, 4), (-1, -1), refine_criteria)
                block_pix = crnrs_refined[4, 0, :]
                break
            else:
                print('checkerboard not found')

            time.sleep(0.01)
        

        # color_im[
        #     int(np.round(block_pix[1])) - 2: int(np.round(block_pix[1])) + 2,
        #     int(np.round(block_pix[0])) - 2: int(np.round(block_pix[0])) + 2
        # ] = 0
        # import imageio
        # imageio.imwrite('check.png', color_im)
        # exit()

        block_z = depth_im[
            int(np.round(block_pix[1])),
            int(np.round(block_pix[0]))
        ]
        block_x = np.multiply(
            block_pix[1] - cam_intr[0, 2],
            block_z / cam_intr[0, 0]
        )
        block_y = np.multiply(
            block_pix[0] - cam_intr[1, 2],
            block_z / cam_intr[1, 1]
        )
        if block_z == 0:
            print('error')
            continue

        # Save calibration point and observed checkerboard center
        observed_pts.append([block_x, block_y, block_z])
        observed_pix.append(block_pix)
        img = dict(color_im=color_im, depth_im=depth_im)
        all_img.append(img)

        # TODO: get measured_pts in world coordiante
        s = input(f'input measured_pts, press Enter to continue.. (x,y separated by space)\n')
      
        words = s.split(' ')

        measured_pt = [float(words[0]), float(words[1]), 0.0]
        c = input(f"Measured :{measured_pt} confirm? (y/n)")
        if c == 'n':
            print("Canceled")
            continue
        else:
            print("Confirmed")
            measured_pts.append(measured_pt)

        # print()
        # measured_pts.append(measured_pt)
        # measured_pts.append(list(map(float, s.split(','))))
    
    all_infos = {
        'all_images': all_img,
        'observed_pts': observed_pts,
        'observed_pix': observed_pix,
        'measured_pts': measured_pts,
    }
    with open('./calibration_images.pkl', 'wb') as f:
        pickle.dump(all_infos, f)


if __name__ == '__main__':
    capture_images(num_calib_pts=12)

    if True:
        with open('./calibration_images.pkl', 'rb') as f:
            all_infos = pickle.load(f)
            observed_pts = np.asarray(all_infos['observed_pts'])
            observed_pix = np.asarray(all_infos['observed_pix'])
            measured_pts = np.asarray(all_infos['measured_pts'])
            world2camera = np.eye(4)
        
        # Estimate rigid transform with SVD (from Nghia Ho)
        def get_rigid_transform(A, B):
            assert len(A) == len(B)
            N = A.shape[0]  # Total points
            centroid_A = np.mean(A, axis=0)
            centroid_B = np.mean(B, axis=0)
            AA = A - np.tile(centroid_A, (N, 1))  # Centre the points
            BB = B - np.tile(centroid_B, (N, 1))
            # Dot is matrix multiplication for array
            H = np.dot(np.transpose(AA), BB)
            U, S, Vt = np.linalg.svd(H)
            R = np.dot(Vt.T, U.T)
            if np.linalg.det(R) < 0:  # Special reflection case
                Vt[2, :] *= -1
                R = np.dot(Vt.T, U.T)
            t = np.dot(-R, centroid_A.T) + centroid_B.T
            return R, t

        def get_rigid_transform_error(z_scale):
            global measured_pts, observed_pts, observed_pix, world2camera

            # Apply z offset and compute new observed points using camera intrinsics
            observed_z = observed_pts[:, 2:] * z_scale
            observed_x = np.multiply(
                observed_pix[:, [0]]-cam_intr[0, 2], observed_z/cam_intr[0, 0])
            observed_y = np.multiply(
                observed_pix[:, [1]]-cam_intr[1, 2], observed_z/cam_intr[1, 1])
            new_observed_pts = np.concatenate(
                (observed_x, observed_y, observed_z), axis=1)

            # Estimate rigid transform between measured points and new observed points
            R, t = get_rigid_transform(np.asarray(
                measured_pts), np.asarray(new_observed_pts))
            t.shape = (3, 1)
            world2camera = np.concatenate(
                (np.concatenate((R, t), axis=1), np.array([[0, 0, 0, 1]])), axis=0)

            # Compute rigid transform error
            registered_pts = np.dot(R, np.transpose(
                measured_pts)) + np.tile(t, (1, measured_pts.shape[0]))
            error = np.transpose(registered_pts) - new_observed_pts
            error = np.sum(np.multiply(error, error))
            rmse = np.sqrt(error/measured_pts.shape[0])
            return rmse

        # Optimize z scale w.r.t. rigid transform error
        print('Calibrating...')
        z_scale_init = 1
        optim_result = optimize.minimize(
            get_rigid_transform_error, np.asarray(z_scale_init), method='Nelder-Mead')
        camera_depth_offset = optim_result.x
        # Save camera optimized offset and camera pose
        print('Saving calibration files...')
        print('camera_depth_offset=', camera_depth_offset)
        np.savetxt('cam_pose/cam2table_depth_scale.txt', camera_depth_offset, delimiter=' ')
        get_rigid_transform_error(camera_depth_offset)
        camera_pose = np.linalg.inv(world2camera)
        np.savetxt('cam_pose/cam2table_pose.txt', camera_pose, delimiter=' ')
        print('Done. Please restart main script.')
