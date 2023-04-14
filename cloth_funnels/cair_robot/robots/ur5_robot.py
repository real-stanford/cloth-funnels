from inspect import trace
import rtde_control
import rtde_receive
import rtde_io
import numpy as np
import time
import threading
from cair_robot.robots.singularity_avoidance import path_avoid_singularity


class UR5RTDE:
    def __init__(self, ip, gripper=None):
        self.rtde_c = rtde_control.RTDEControlInterface(ip)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(ip)
        self.rtde_i = None

        if gripper == 'rg2':
            self.rtde_i = rtde_io.RTDEIOInterface(ip)
        self.gripper = gripper
        self.home_joint = [np.pi/2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0]
        if self.gripper is None:
            self.rtde_c.setTcp([0, 0, 0, 0, 0, 0])
        elif gripper == 'rg2':
            self.rtde_c.setTcp([0, 0, 0.195, 0, 0, 0])
            self.rtde_c.setPayload(1.043, [0, 0, 0.08])
        else:
            self.rtde_c.setTcp(self.gripper.tool_offset)
            self.rtde_c.setPayload(self.gripper.mass, [0, 0, 0.08])
    
    def __del__(self):
        self.disconnect()
    
    def disconnect(self):
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
        if hasattr(self.gripper, 'disconnect'):
            self.gripper.disconnect()

    def home(self, speed=1.5, acceleration=1, blocking=True):
        return self.rtde_c.moveJ(self.home_joint, speed, acceleration, not blocking)

    def movej(self, q, speed=1.5, acceleration=1, blocking=True):
        return self.rtde_c.moveJ(q, speed, acceleration, not blocking)
    
    def movel(self, p, speed=1.5, acceleration=1, blocking=True, avoid_singularity=False):
        # nomralize input format to 2D numpy array
        if not isinstance(p, np.ndarray):
            p = np.array(p)
        if len(p.shape) == 1:
            p = p.reshape(1,-1)
        
        if avoid_singularity:
            path = np.concatenate([
                self.get_tcp_pose().reshape(-1,6),
                p],axis=0)
            new_path = path_avoid_singularity(path)
            p = new_path[1:]

        if p.shape[0] == 1:
            return self.rtde_c.moveL(p[0].tolist(), speed, acceleration, not blocking)
        else:
            p = p.tolist()
            for x in p:
                x.extend([speed, acceleration, 0])
            return self.rtde_c.moveL(p, not blocking)
    
    def movej_ik(self, p, speed=1.5, acceleration=1, blocking=True):
        return self.rtde_c.moveJ_IK(p, speed, acceleration, not blocking)

    def open_gripper(self, sleep_time=1):
        if self.gripper == 'rg2':
            r = self.rtde_i.setToolDigitalOut(0, False)
        else:
            self.gripper.open(blocking=True)
            r = True
        time.sleep(sleep_time)
        return r
        
    def close_gripper(self, sleep_time=1):
        if self.gripper == 'rg2':
            r = self.rtde_i.setToolDigitalOut(0, True)
        else:
            self.gripper.close(blocking=False)
            r = True
        time.sleep(sleep_time)
        return r
    
    def start_force_mode(self):
        class ForceModeGuard:
            def __init__(self, rtde_c):
                self.rtde_c = rtde_c
                self.enabled = False
            
            def __enter__(self):
                self.enabled=True
                return self
            
            def __exit__(self, type, value, traceback):
                if value is not None:
                    print(value, traceback)
                self.rtde_c.forceModeStop()
                self.enabled = False
                return True

            def apply_force(self, task_frame, selection_vector, wrench, type, limits):
                if not self.enabled:
                    return False
                return self.rtde_c.forceMode(task_frame, selection_vector, wrench, type, limits)
        return ForceModeGuard(self.rtde_c)

    def get_tcp_pose(self):
        return np.array(self.rtde_r.getActualTCPPose())
    
    def get_tcp_speed(self):
        return np.array(self.rtde_r.getActualTCPSpeed())

    def get_tcp_force(self):
        return np.array(self.rtde_r.getActualTCPForce())

class UR5PairRTDE:
    def __init__(self, left_ur5, right_ur5):
        self.left_ur5 = left_ur5
        self.right_ur5 = right_ur5

    def home(self, speed=1.5, acceleration=1, blocking=True):
        t1 = threading.Thread(target=self.left_ur5.home, args=(speed, acceleration, blocking))
        t2 = threading.Thread(target=self.right_ur5.home, args=(speed, acceleration, blocking))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    def movej(self, q_left, q_right, speed=1.5, acceleration=1, blocking=True):
        t1 = threading.Thread(target=self.left_ur5.movej, args=(q_left, speed, acceleration, blocking))
        t2 = threading.Thread(target=self.right_ur5.movej, args=(q_right, speed, acceleration, blocking))
        t1.start()
        t2.start()
        t1.join()
        t2.join()


    def movej_ik(self, p_left, p_right, speed=1.5, acceleration=1, blocking=True):
        t1 = threading.Thread(target=self.left_ur5.movej_ik, args=(p_left, speed, acceleration, blocking))
        t2 = threading.Thread(target=self.right_ur5.movej_ik, args=(p_right, speed, acceleration, blocking))
        t1.start()
        t2.start()
        t1.join()
        t2.join()


    def movel(self, p_left, p_right, speed=1.5, acceleration=1, blocking=True):
        t1 = threading.Thread(target=self.left_ur5.movel, args=(p_left, speed, acceleration, blocking))
        t2 = threading.Thread(target=self.right_ur5.movel, args=(p_right, speed, acceleration, blocking))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    def open_gripper(self, sleep_time=1):
        self.left_ur5.open_gripper(0)
        self.right_ur5.open_gripper(0)
        time.sleep(sleep_time)

    def close_gripper(self, sleep_time=1):
        self.left_ur5.close_gripper(0)
        self.right_ur5.close_gripper(0)
        time.sleep(sleep_time)
