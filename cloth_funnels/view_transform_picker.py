from typing import Tuple
import numpy as np
import cv2
from cair_robot.cameras.kinect_client import KinectClient
from skimage.transform import warp, AffineTransform
from cair_robot.common.geometry import get_center_affine
import time
import argparse

class TransformLabeler:
    """Function object for GUI labeling"""
    def __init__(self, img_size=128):
        """
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        """
        self.coord = None
        self.x = 579
        self.y = 275
        self.trans_delta = 2
        self.angle = 0
        self.angle_delta = np.pi / 64
        self.scale = 4.24
        self.scale_delta = 0.1
        self.img_size = img_size
    
    def callback(self, action, x, y, flags, *userdata):
        """
        https://docs.opencv.org/4.5.5/d7/dfc/group__highgui.html#gab7aed186e151d5222ef97192912127a4
        """
        if action == cv2.EVENT_LBUTTONUP:
            print('callback')
            coord = (x,y)
            self.coord = coord
    
    def get_transform(self, img: np.ndarray
            , from_args=False, angle=0, x=579, y=275, scale=4.24, **kwargs) -> Tuple[Tuple[int, int], float]:
        """
        Main loop for GUI.
        :img: RGB observation
        :coord: tuple(int x, int y). By OpenCV convension, x is left-to-right and y is top-to-bottom.
        :angle: float. By OpenCV convension, angle is clockwise rotation.
        """
        if not from_args:
            cv2.namedWindow("raw_img", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("raw_img", self.callback)
            cv2.namedWindow("result_img", cv2.WINDOW_NORMAL)

        corner_coords = np.array([
            [0,0],
            [0,1],
            [1,1],
            [1,0]
        ], dtype=np.float32) * self.img_size
        arrow_coords = np.array([
            [0.5,0.5],
            [0.7,0.5]
        ], dtype=np.float32) * self.img_size

        vis_img = img.copy()
        prev_t = time.perf_counter()
        tf = None
        while True:
            vis_img[:] = img[:]

            # tf = AffineTransform(
            #     scale=self.scale, 
            #     rotation=self.angle, 
            #     translation=[self.x,self.y])
            if from_args:
                self.scale = scale
                self.angle = angle
                self.x = x
                self.y = y
                
            tf = get_center_affine(
                img_shape=(self.img_size, self.img_size),
                scale=self.scale, 
                rotation=self.angle, 
                translation=[self.x,self.y])

            if from_args: break
                
            # use non-inverted transform for user experience
            tf_img = warp(img, tf)
            result_img = tf_img[:self.img_size,:self.img_size]
            cv2.imshow("result_img", result_img[:,:,[2,1,0]])

            draw_coords = tf(corner_coords).astype(np.int32)
            draw_arrow_coords = tf(arrow_coords).astype(np.int32)
            cv2.polylines(vis_img, [draw_coords],True,(0,0,255))
            cv2.drawMarker(vis_img, draw_coords[0],(0,255,0))
            cv2.drawMarker(vis_img, draw_arrow_coords[0],(255,0,0))
            cv2.polylines(vis_img, [draw_arrow_coords],False,(255,0,0))

            cv2.imshow("raw_img", vis_img[:,:,[2,1,0]])

            curr_t = time.perf_counter()
            diff_t = curr_t - prev_t
            print('FPS: {:.1f}'.format(1/diff_t))
            prev_t = curr_t
            delay = max(0, 1/60-diff_t)
            key = cv2.waitKey(delay)
            if key == ord('q'):
                self.angle -= self.angle_delta
            elif key == ord('e'):
                self.angle += self.angle_delta
            elif key == ord('a'):
                self.x -= self.trans_delta
            elif key == ord('d'):
                self.x += self.trans_delta
            elif key == ord('w'):
                self.y -= self.trans_delta
            elif key == ord('s'):
                self.y += self.trans_delta
            elif key == ord('z'):
                self.scale -= self.scale_delta
            elif key == ord('c'):
                self.scale += self.scale_delta
            elif key == 13:
                print('angle: {:.2f}'.format(self.angle))
                print('x: {:.2f}'.format(self.x))
                print('y: {:.2f}'.format(self.y))
                print('scale: {:.2f}'.format(self.scale))
                # print('enter')
                break
        
        tx_camera_img = np.linalg.inv(tf.params)
        return tx_camera_img


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--from_args', default=False, action='store_true')
    parser.add_argument('--angle', default=0, type=float)
    parser.add_argument('--x', default=579, type=float)
    parser.add_argument('--y', default=275, type=float)
    parser.add_argument('--scale', default=4.24, type=float)

    args = parser.parse_args()
    # print("args:", args)

    
    camera = KinectClient('128.59.23.32', '8080', fielt_bg=False)
    color, depth = camera.get_camera_data()
    labeler = TransformLabeler()
    # from img to camera
    if not args.from_args:
        tx_camera_img = labeler.get_transform(color)
    else:
        tx_camera_img = labeler.get_transform(color, **vars(args))


    print(tx_camera_img)
    print(labeler.angle)
    print(labeler.scale)
    print(labeler.x, labeler.y)
    np.savetxt('cam_pose/view2cam.txt', tx_camera_img)

if __name__ == '__main__':
    main()
