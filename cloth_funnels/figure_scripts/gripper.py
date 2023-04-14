#!/usr/bin/env python

import time
import threading

import numpy as np
import pybullet as p


class Gripper:

    def __init__(self):
        self.activated = False

    def step(self):
        return

    def activate(self, objects):
        return

    def release(self):
        return

# -----------------------------------------------------------------------------
# Suction-Based Gripper
# -----------------------------------------------------------------------------


class Suction(Gripper):

    def __init__(self, robot_id, tool_link, max_force=50):
        position = (0.487, 0.109, 0.351)
        rotation = p.getQuaternionFromEuler((np.pi, 0, 0))
        urdf = 'assets/ur5/suction/suction-head.urdf'
        self.body = p.loadURDF(urdf, position, rotation)

        constraint_id = p.createConstraint(
            parentBodyUniqueId=robot_id,
            parentLinkIndex=tool_link,
            childBodyUniqueId=self.body,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.07))
        p.changeConstraint(constraint_id, maxForce=max_force)

        self.activated = False
        self.contact_constraint = None

    def activate(self, possible_objects):
        """
        Simulates suction by creating rigid fixed constraint between suction
        gripper and contacted object.
        """
        if not self.activated:
            # Only report contact points involving linkIndexA of bodyA (the
            # suction) -- returns a list (actually, a tuple) of such points.
            points = p.getContactPoints(bodyA=self.body, linkIndexA=0)

            print('inside gripper.activate(), points: {} (len: {}), possible_objects: {}'.format(
                points, len(points), possible_objects))

            if len(points) > 0:
                for point in points:
                    object_id, contact_link = point[2], point[4]
                if object_id in possible_objects:
                    body_pose = p.getLinkState(self.body, 0)
                    object_pose = p.getBasePositionAndOrientation(object_id)
                    world_to_body = p.invertTransform(
                        body_pose[0], body_pose[1])
                    object_to_body = p.multiplyTransforms(
                        world_to_body[0], world_to_body[1],
                        object_pose[0], object_pose[1])
                    self.contact_constraint = p.createConstraint(
                        parentBodyUniqueId=self.body,
                        parentLinkIndex=0,
                        childBodyUniqueId=object_id,
                        childLinkIndex=contact_link,
                        jointType=p.JOINT_FIXED,
                        jointAxis=(0, 0, 0),
                        parentFramePosition=object_to_body[0],
                        parentFrameOrientation=object_to_body[1],
                        childFramePosition=(0, 0, 0),
                        childFrameOrientation=(0, 0, 0))
            self.activated = True

    def release(self):
        """
        If suction off, detect contact between gripper and objects.
        If suction on, detect contact between picked object and other objects.
        """
        if self.activated:
            if self.contact_constraint is not None:
                try:
                    p.removeConstraint(self.contact_constraint)
                    self.contact_constraint = None
                except:
                    pass
            self.activated = False

    def detect_contact(self):
        """
        If suction off, detect contact between gripper and objects.
        If suction on, detect contact between picked object and other objects.
        """
        body, link = self.body, 0
        if self.activated and self.contact_constraint is not None:
            try:
                info = p.getConstraintInfo(self.contact_constraint)
                body, link = info[2], info[3]
            except:
                self.contact_constraint = None
                pass
        points = p.getContactPoints(bodyA=body, linkIndexA=link)
        if self.activated:
            points = [point for point in points if point[2] != self.body]
        if len(points) > 0:
            print('inside gripper.detect_contact(), points: {} (len: {})'.format(
                points, len(points)))
        return len(points) > 0

    def check_grasp(self):
        # Index 2 in getConstraintInfo returns childBodyUniqueId.
        suctioned_object = None
        if not self.contact_constraint is None:
            suctioned_object = p.getConstraintInfo(self.contact_constraint)[2]
        return not self.contact_constraint is None

# -----------------------------------------------------------------------------
# Parallel-Jaw Two-Finger Gripper (TODO: fix)
# -----------------------------------------------------------------------------


class Robotiq2F85:

    def __init__(self, robot, tool):
        self.robot = robot
        self.tool = tool
        pos = [0.487, 0.109, 0.421]
        rot = p.getQuaternionFromEuler([np.pi, 0, 0])
        urdf = 'assets/ur5/gripper/robotiq_2f_85.urdf'
        self.body = p.loadURDF(urdf, pos, rot)
        self.n_joints = p.getNumJoints(self.body)
        self.activated = False

        # Connect gripper base to robot tool
        p.createConstraint(self.robot, tool, self.body, 0,
                           jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
                           parentFramePosition=[0, 0, 0],
                           childFramePosition=[0, 0, -0.05])

        # Set friction coefficients for gripper fingers
        for i in range(p.getNumJoints(self.body)):
            p.changeDynamics(self.body, i,
                             lateralFriction=1.5,
                             spinningFriction=1.0,
                             rollingFriction=0.0001,
                             # rollingFriction=1.0,
                             frictionAnchor=True)  # contactStiffness=0.0, contactDamping=0.0

        # Start thread to handle additional gripper constraints
        self.motor_joint = 1
        # self.constraints_thread = threading.Thread(target=self.step)
        # self.constraints_thread.daemon = True
        # self.constraints_thread.start()

    # Control joint positions by enforcing hard contraints on gripper behavior
    # Set one joint as the open/close motor joint (other joints should mimic)
    def step(self):
        # while True:
        currj = [p.getJointState(self.body, i)[0]
                 for i in range(self.n_joints)]
        indj = [6, 3, 8, 5, 10]
        targj = [currj[1], -currj[1], -currj[1], currj[1], currj[1]]
        p.setJointMotorControlArray(self.body, indj, p.POSITION_CONTROL,
                                    targj, positionGains=np.ones(5))
        # time.sleep(0.001)

    # Close gripper fingers and check grasp success (width between fingers
    # exceeds some threshold)
    def activate(self, valid_obj=None):
        p.setJointMotorControl2(self.body, self.motor_joint,
                                p.VELOCITY_CONTROL, targetVelocity=1, force=100)
        if not self.external_contact():
            while self.moving():
                time.sleep(0.001)
        self.activated = True

    # Open gripper fingers
    def release(self):
        p.setJointMotorControl2(self.body, self.motor_joint,
                                p.VELOCITY_CONTROL, targetVelocity=-1, force=100)
        while self.moving():
            time.sleep(0.001)
        self.activated = False

    # If activated and object in gripper: check object contact
    # If activated and nothing in gripper: check gripper contact
    # If released: check proximity to surface
    def detect_contact(self):
        obj, link, ray_frac = self.check_proximity()
        if self.activated:
            empty = self.grasp_width() < 0.01
            cbody = self.body if empty else obj
            if obj == self.body or obj == 0:
                return False
            return self.external_contact(cbody)
        else:
            return ray_frac < 0.14 or self.external_contact()

    # Return if body is in contact with something other than gripper
    def external_contact(self, body=None):
        if body is None:
            body = self.body
        pts = p.getContactPoints(bodyA=body)
        pts = [pt for pt in pts if pt[2] != self.body]
        return len(pts) > 0

    # Check grasp success
    def check_grasp(self):
        while self.moving():
            time.sleep(0.001)
        success = self.grasp_width() > 0.01
        return success

    def grasp_width(self):
        lpad = np.array(p.getLinkState(self.body, 4)[0])
        rpad = np.array(p.getLinkState(self.body, 9)[0])
        dist = np.linalg.norm(lpad - rpad) - 0.047813
        return dist

    # Helper functions

    def moving(self):
        v = [np.linalg.norm(p.getLinkState(
            self.body, i, computeLinkVelocity=1)[6]) for i in [3, 8]]
        return any(np.array(v) > 1e-2)

    def check_proximity(self):
        ee_pos = np.array(p.getLinkState(self.robot, self.tool)[0])
        tool_pos = np.array(p.getLinkState(self.body, 0)[0])
        vec = (tool_pos - ee_pos) / np.linalg.norm((tool_pos - ee_pos))
        ee_targ = ee_pos + vec
        ray_data = p.rayTest(ee_pos, ee_targ)[0]
        obj, link, ray_frac = ray_data[0], ray_data[1], ray_data[2]
        return obj, link, ray_frac
