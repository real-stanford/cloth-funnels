

import pickle
import sys
import pybullet as p
import pybullet_data
from ur5 import UR5RobotiqPybulletController as UR5
import PySimpleGUI as sg
import pickle
from os import getcwd
from urdfpy import URDF
from os.path import abspath, dirname, basename, splitext
from transforms3d.affines import decompose
from transforms3d.quaternions import mat2quat
from time import sleep
import numpy as np
import quaternion
from threading import Thread


class PybulletRecorder:
    class LinkTracker:
        def __init__(self,
                     name,
                     id,
                     link_id,
                     link_origin,
                     mesh_path,
                     mesh_scale):
            self.id = id
            self.link_id = link_id
            self.mesh_path = mesh_path
            self.mesh_scale = mesh_scale
            decomposed_origin = decompose(link_origin)
            orn = mat2quat(decomposed_origin[1])
            orn = [orn[1], orn[2], orn[3], orn[0]]
            self.link_pose = [decomposed_origin[0], orn]
            self.name = name

        def transform(self, position, orientation):
            return p.multiplyTransforms(
                position, orientation,
                self.link_pose[0], self.link_pose[1])

        def get_keyframe(self):
            if self.link_id == -1:
                position, orientation = p.getBasePositionAndOrientation(
                    self.id)
                position, orientation = self.transform(
                    position=position, orientation=orientation)
            else:
                link_state = p.getLinkState(self.id,
                                            self.link_id,
                                            computeForwardKinematics=True)
                position, orientation = self.transform(
                    position=link_state[4],
                    orientation=link_state[5])
            return {
                'position': list(position),
                'orientation': list(orientation)
            }

    def __init__(self):
        self.states = []
        self.links = []

    def register_object(self, id, path):
        full_path = abspath(path)
        dir_path = dirname(full_path)
        file_name = splitext(basename(path))[0]
        robot = URDF.load(path)
        num_joints = p.getNumJoints(id)
        assert num_joints + 1 == len(robot.links)
        if len(robot.links) == 1:
            self.links.append(
                PybulletRecorder.LinkTracker(
                    name=file_name + f'_{id}_root',
                    id=id,
                    link_id=-1,
                    link_origin=robot.links[0].visuals[0].origin,
                    mesh_path=dir_path + '/' +
                    robot.links[0].visuals[0].geometry.mesh.filename,
                    mesh_scale=robot.links[0].visuals[0].geometry.mesh.scale))
        else:
            self.links.append(
                PybulletRecorder.LinkTracker(
                    name=file_name + f'_{id}_root',
                    id=id,
                    link_id=-1,
                    link_origin=robot.links[0].visuals[0].origin,
                    mesh_path=dir_path + '/' +
                    robot.links[0].visuals[0].geometry.mesh.filename,
                    mesh_scale=robot.links[0].visuals[0].geometry.mesh.scale))
            for link_id, link in enumerate(robot.links):
                if len(link.visuals) > 0:
                    if link_id > 7:
                        link_id -= 3
                    if p.getLinkState(id, link_id) is not None\
                            and link.visuals[0].geometry.mesh:
                        # hard code for robotiq
                        self.links.append(
                            PybulletRecorder.LinkTracker(
                                name=file_name + f'_{id}_{link.name}',
                                id=id,
                                link_id=link_id,
                                link_origin=link.visuals[0].origin,
                                mesh_path=dir_path + '/' +
                                link.visuals[0].geometry.mesh.filename,
                                mesh_scale=link.visuals[0].geometry.mesh.scale))

    def add_keyframe(self):
        # Ideally, call every stepSimulation
        current_state = {}
        for link in self.links:
            current_state[link.name] = link.get_keyframe()
        self.states.append(current_state)

    def prompt_save(self):
        layout = [[sg.Text('Do you want to save previous episode?')],
                  [sg.Button('Yes'), sg.Button('No')]]
        window = sg.Window('PyBullet Recorder', layout)
        save = False
        while True:
            event, values = window.read()
            if event in (None, 'No'):
                break
            elif event == 'Yes':
                save = True
                break
        window.close()
        if save:
            layout = [[sg.Text('Where do you want to save it?')],
                      [sg.Text('Path'), sg.InputText(getcwd())],
                      [sg.Button('OK')]]
            window = sg.Window('PyBullet Recorder', layout)
            event, values = window.read()
            window.close()
            self.save(values[0])
        self.reset()

    def reset(self):
        self.states = []

    def get_formatted_output(self):
        retval = {}
        for link in self.links:
            retval[link.name] = {
                'type': 'mesh',
                'mesh_path': link.mesh_path,
                'mesh_scale': link.mesh_scale,
                'frames': [state[link.name] for state in self.states]
            }
        return retval

    def save(self, path):
        if path is None:
            print("[Recorder] Path is None.. not saving")
        else:
            print("[Recorder] Saving state to {}".format(path))
            pickle.dump(self.get_formatted_output(), open(path, 'wb'))


if __name__ == '__main__':
    gripper_states = pickle.load(open(sys.argv[-1], 'rb'))
    output_path = sys.argv[-1].split('.pkl')[0] + '_ur5.pkl'
    gui = True
    p.connect(p.GUI if gui else p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.0, cameraYaw=0, cameraPitch=-10,
        cameraTargetPosition=(0, 0, 0.2))
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setRealTimeSimulation(False)
    p.setGravity(0, 0, -9.8)
    p.loadURDF(fileName="plane.urdf",
               basePosition=(0, 0, -0.01))
    dist = 1.35
    left_ur5 = UR5(
        base_pose=[[-0.2, dist/2, 0.1],
                   p.getQuaternionFromEuler((0, 0, np.pi/2))])
    right_ur5 = UR5(
        base_pose=[[-0.2, -dist/2, 0.1],
                   p.getQuaternionFromEuler((0, 0, -np.pi/2))])
    recorder = PybulletRecorder()
    recorder.register_object(
        id=right_ur5.id, path='assets/figures-ur5/ur5_robotiq.urdf')
    recorder.register_object(
        id=left_ur5.id, path='assets/figures-ur5/ur5_robotiq.urdf')
    gripper_size = 0.11
    straight_orn = p.getQuaternionFromEuler((0, np.pi/2, 0))
    # straight_orn = p.getQuaternionFromEuler((np.pi/2, np.pi/2, np.pi/2))
    for i, (left_pos, right_pos) in enumerate(gripper_states):
        left_pos = [left_pos[2], left_pos[0], left_pos[1]+gripper_size]
        right_pos = [right_pos[2], right_pos[0], right_pos[1]+gripper_size]
        if i > 3 and i < 220:
            right_ur5.set_gripper_joints(right_ur5.CLOSED_POSITION)
            left_ur5.set_gripper_joints(right_ur5.CLOSED_POSITION)
        else:
            right_ur5.set_gripper_joints(right_ur5.OPEN_POSITION)
            left_ur5.set_gripper_joints(right_ur5.OPEN_POSITION)
        targ_orn = straight_orn
        left_ur5.set_arm_joints(left_ur5.initial_arm_joint_values)
        targ_j, pos_err, rot_err = left_ur5.get_arm_ik_pybullet(
            [left_pos, targ_orn])
        left_ur5.set_arm_joints(targ_j)
        right_ur5.set_arm_joints(right_ur5.initial_arm_joint_values)
        targ_j, pos_err, rot_err = right_ur5.get_arm_ik_pybullet(
            [right_pos, targ_orn])
        right_ur5.set_arm_joints(targ_j)
        p.stepSimulation()
        recorder.add_keyframe()
        if gui:
            sleep(0.1)
    recorder.save(output_path)
