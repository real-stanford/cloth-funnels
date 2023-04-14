

from real.realsense import RealSense

def get_front_cam():
    return RealSense(
        tcp_ip='127.0.0.1',
        tcp_port=50010,
        im_h=720,
        im_w=1280,
        max_depth=3.0)

def get_top_cam():
    return RealSense(
        tcp_ip='127.0.0.1',
        tcp_port=50011,
        im_h=720,
        im_w=1280,
        max_depth=3.0)

