from dm_control import mjcf
from dm_control import composer
from dm_control import mujoco
from dm_control import viewer
from dm_control.composer import variation
from dm_control.composer.variation import distributions
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import floors
from src.controllers import RandomController
from dm_env import StepType

import numpy as np
import matplotlib.pyplot as plt

seed = 0
# np.random.seed(seed)


def check_body_collision(physics: mjcf.Physics, body1: str, body2: str):
    """
    Check whether the given two bodies have collision. The given body name can be either child body or parent body
    *NOTICE*: this method may be unsafe and cause hidden bug, since the way of retrieving body is to check whether
    the given name is a sub-string of the collision body
    :param physics: a MuJoCo physics engine
    :param body1: the name of body1
    :param body2: the name of body2
    :return collision: a bool variable
    """
    collision = False
    for geom1, geom2 in zip(physics.data.contact.geom1, physics.data.contact.geom2):
        bodyid1 = physics.model.geom_bodyid[geom1]
        bodyid2 = physics.model.geom_bodyid[geom2]
        bodyname1 = physics.model.id2name(bodyid1, 'body')
        bodyname2 = physics.model.id2name(bodyid2, 'body')
        if (body1 in bodyname1 and body2 in bodyname2) or (body2 in bodyname1 and body1 in bodyname2):
            collision = True
            break
    return collision


class RopeEntity(composer.Entity):
    def _build(self, length=20, rgba=(0.2, 0.8, 0.2, 1), friction_noise=0.):
        self._model = mjcf.RootElement('rope')
        self._model.compiler.angle = 'radian'
        # self._model.default.geom.friction = [1, 0.1, 0.1]
        body = self._model.worldbody.add('body', name='rB0')
        self._composite = body.add('composite', prefix="r", type='rope', count=[length, 1, 1], spacing=0.084)
        self._composite.add('joint', kind='main', damping=0.01, stiffness=0.01)
        self._composite.geom.set_attributes(type='capsule', size=[0.0375, 0.04], rgba=rgba, mass=0.005,
                                            contype=1, conaffinity=1, priority=1, friction=[0.1, 5e-3, 1e-4])
        self._friction_noise_generator = distributions.LogNormal(sigma=[friction_noise for _ in range(3)])
        self.reset_friction()

    @property
    def mjcf_model(self):
        return self._model

    def reset_friction(self, random_state=None):
        friction_noise = self._friction_noise_generator(random_state=random_state)
        new_friction = np.array([1, 5e-3, 1e-4]) * friction_noise
        self._composite.geom.set_attributes(friction=new_friction)

    def reset_pose(self):
        pass


class CupEntity(composer.Entity):
    def _build(self, name=None):
        self._model = mjcf.RootElement(name)
        body = self._model.worldbody.add('body', name='dummy')
        # Cup Slider x
        body.add('joint', axis=[1, 0, 0], type='slide', name='x', pos=[0, 0, 0], limited=True, range=[-40.0, 40.0])
        # Cup Base
        body.add('geom', type="capsule", pos="0 0 0", quat="0.707 0 0.707 0", size="0.05 0.1625", rgba="1 0 0 1", mass="0.5", contype="2",
                 conaffinity="2")
        # Cup left section 1
        body.add('geom', type="capsule", fromto="0.1625 0 0 0.225 0 0.075", size="0.05", rgba="1 0 0 1", mass=".25", contype="2", conaffinity="2")
        # Cup right section 1
        body.add('geom', type="capsule", fromto="-0.1625 0 0 -0.225 0 0.075", size="0.05", rgba="1 0 0 1", mass=".25",
                 contype="2", conaffinity="2")
        # Cup left section 2
        body.add('geom', type="capsule", fromto="0.225 0 0.075 0.225 0 0.25", size="0.05", rgba="1 0 0 1", mass=".25", contype="2", conaffinity="2")
        # Cup right section 2
        body.add('geom', type="capsule", fromto="-0.225 0 0.075 -0.225 0 0.25", size="0.05", rgba="1 0 0 1", mass=".25",
                 contype="2", conaffinity="2")

    @property
    def mjcf_model(self):
        return self._model


class RopeManipulation(composer.Task):
    # The number of physics substeps per control timestep. Default physics substep takes 1ms.
    #  action repition ... ?
    NUM_SUBSTEPS = 100

    def __init__(self, rope_length=11, action_noise: float = 0.0, friction_noise: float = 0.2):
        # root entity
        self._arena = composer.Arena()

        # simulation setting
        self._arena.mjcf_model.compiler.inertiafromgeom = True
        self._arena.mjcf_model.default.joint.damping = 0
        self._arena.mjcf_model.default.joint.stiffness = 0
        self._arena.mjcf_model.default.geom.contype = 3
        self._arena.mjcf_model.default.geom.conaffinity = 3
        self._arena.mjcf_model.default.geom.friction = [1, 0.1, 0.1]
        self._arena.mjcf_model.option.gravity = [1e-5, 0, -9.81]
        self._arena.mjcf_model.option.integrator = 'Euler'    # RK4 or Euler
        self._arena.mjcf_model.option.timestep = 0.001

        # other entities
        self._rope = RopeEntity(length=rope_length, friction_noise=friction_noise)
        self._cup = CupEntity(name='cup')

        rope_site = self._arena.add_free_entity(self._rope)
        cup_site = self._arena.attach(self._cup)

        rope_site.pos = [0.0, 5.0, 0.0]
        cup_site.pos = [0.0, 5.0, 1.0]

        # constraint
        self._arena.mjcf_model.equality.add('connect', body1='cup/dummy', body2='rope/rB0', anchor=[-1.0, 0, 0.0])

        # noise
        self.action_noise = action_noise
        self.friction_noise = friction_noise

        # texture and light
        self._arena.mjcf_model.worldbody.add('light', pos=[0, 0, 3], dir=[0, 0, -1])

        # camera
        top_camera = self._arena.mjcf_model.worldbody.add(
            'camera',
            name='top_camera',
            pos=[0, 0, 10],
            quat=[1, 0, 0, 0],
            fovy=np.rad2deg(3.0))

        # top_camera = self._arena.mjcf_model.worldbody('camera')[0]
        # top_camera.pos = [0, 0, 20]
        env_camera = self._arena.mjcf_model.worldbody.add('camera', name='env_camera', fovy=top_camera.fovy, pos=[5, 0, 20])

        # actuators
        self._arena.mjcf_model.actuator.add('position', name='left_x', joint=self._cup.mjcf_model.find_all('joint')[0],
                                            ctrllimited=True, ctrlrange=[-2, 2], kp=0.5)
        # self._arena.mjcf_model.actuator.add('position', name='left_y', joint=self._gripper1.mjcf_model.find_all('joint')[1],
        #                                     ctrllimited=True, ctrlrange=[-2, 2], kp=0.5)
        # self._arena.mjcf_model.actuator.add('position', name='left_z', joint=self._gripper1.mjcf_model.find_all('joint')[2],
        #                                     ctrllimited=True, ctrlrange=[-2, 2], kp=0.5)
        # self._arena.mjcf_model.actuator.add('motor', name='left_x', joint=self._gripper1.mjcf_model.find_all('joint')[0],
        #                                     forcelimited=True, forcerange=[-20, 20])
        # self._arena.mjcf_model.actuator.add('motor', name='left_y', joint=self._gripper1.mjcf_model.find_all('joint')[1],
        #                                     forcelimited=True, forcerange=[-20, 20])
        # self._arena.mjcf_model.actuator.add('motor', name='left_z', joint=self._gripper1.mjcf_model.find_all('joint')[2],
        #                                     forcelimited=True, forcerange=[-20, 20])
        self._actuators = self._arena.mjcf_model.find_all('actuator')

        # Configure initial poses
        self._xy_range = distributions.Uniform(-1., 1.)
        self._joint_range = distributions.Uniform(-1, 1)
        # self._x_range = distributions.Uniform(-0.7, 0.7)
        # self._y_range = distributions.Uniform(-0.7, 0.7)
        # self._rope_initial_pose = UniformBox(self._x_range, self._y_range)
        # self._vx_range = distributions.Uniform(-3, 3)
        # self._vy_range = distributions.Uniform(-3, 3)
        # self._rope_initial_velocity = UniformBox(self._x_range, self._y_range)
        # self._goal_x_range = distributions.Uniform(-0.7, 0.7)
        # self._goal_y_range = distributions.Uniform(-0.7, 0.7)
        # self._goal_generator = UniformBox(self._goal_x_range, self._goal_y_range)

        # Configure variators (for randomness)
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        self._task_observables = {}
        self._task_observables['rope_pos'] = observable.MujocoFeature('geom_xpos', [f'rope/rG{i}' for i in range(rope_length)])

        # Configure and enable observables
        # pos_corruptor = noises.Additive(distributions.Normal(scale=0.01))
        # pos_corruptor = None
        # self._task_observables['robot_position'].corruptor = pos_corruptor
        # self._task_observables['robot_position'].enabled = True
        # vel_corruptor = noises.Multiplicative(distributions.LogNormal(sigma=0.01))
        # vel_corruptor = None
        # self._task_observables['robot_velocity'].corruptor = vel_corruptor
        # self._task_observables['robot_velocity'].enabled = True

        for obs in self._task_observables.values():
            obs.enabled = True

        self.control_timestep = self.NUM_SUBSTEPS * self.physics_timestep

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode_mjcf(self, random_state):
        # random_state = np.random.RandomState(0)
        self._rope.reset_friction()

    def initialize_episode(self, physics, random_state):
        # pass
        # random_state = np.random.RandomState(0)
        while True:
            x, y = self._xy_range(initial_value=np.zeros(2), random_state=random_state)
            joints = self._joint_range(initial_value=np.zeros(10), random_state=random_state)
            with physics.reset_context():
                self._rope.set_pose(physics, position=(x, y, 0.0375))
                # physics.bind(self._gripper1.mjcf_model.find_all('joint')).qpos = np.array([x-0.075, y, 0])
                self._cup.set_pose(physics, position=(x-0.075, y, 0.0375))
                for i in range(10):
                    physics.named.data.qpos[f'rope/rJ1_{i+1}'] = joints[i]
            if check_body_collision(physics, 'maze', 'rope') or check_body_collision(physics, 'maze', 'gripper'):
                continue
            else:
                break

    def before_step(self, physics, action, random_state):
        action_noise = distributions.Normal(scale=self.action_noise)
        action = action + action_noise(random_state=random_state)
        physics.set_control(action)

    def after_step(self, physics, random_state):
        pass
        # print(check_body_collision(physics, 'unnamed_model/', 'unnamed_model_1/'))
        # robot_pos = physics.bind(self.rob_freejoint).qpos
        # robot_vel = physics.bind(self.rob_freejoint).qvel
        # original_rope_pos = robot_pos.copy()
        # original_rope_vel = robot_vel.copy()
        # pos_noise = distributions.Normal(scale=self.process_noise)
        # vel_noise = distributions.LogNormal(sigma=self.process_noise)
        # robot_pos[0:2] = robot_pos[0:2] + pos_noise(random_state=random_state)
        # robot_vel[0:2] = robot_vel[0:2] * vel_noise(random_state=random_state)

    def get_reward(self, physics):
        # return self._button.num_activated_steps / NUM_SUBSTEPS
        # collision = check_body_collision(physics, 'unnamed_model/', 'unnamed_model_1/')
        # return (collision,)
        return 0

    # def get_aerial_view(self, physics, show=False) -> np.ndarray:
    #     # pass
    #     # # move the robot and the goal out of camera view temporarily
    #     # origin_qpos = physics.data.qpos.copy()
    #     # origin_qvel = physics.data.qvel.copy()
    #     # # origin_rope_pos = physics.bind(mjcf.get_frame_freejoint(self._rope.mjcf_model)).qpos.copy()
    #     # # origin_rope_vel = physics.bind(mjcf.get_frame_freejoint(self._rope.mjcf_model)).qvel.copy()
    #     # origin_gripper_pos = physics.bind(mjcf.get_attachment_frame(self._gripper1.mjcf_model)).pos.copy()
    #     # # origin_gripper_pos = physics.bind(self._gripper1.mjcf_model.find_all('joint')).qpos.copy()
    #     # # origin_gripper_vel = physics.bind(self._gripper1.mjcf_model.find_all('joint')).qvel.copy()
    #     # with physics.reset_context():
    #     #     self._rope.set_pose(physics, position=[999, 999, 10])
    #     #     self._gripper1.set_pose(physics, position=[999-0.075, 999, 10])
    #     #     # physics.bind(self._gripper1.mjcf_model.find_all('joint')).qpos = np.array([999, 999, 10])
    #     camera = mujoco.Camera(physics, height=128, width=128, camera_id='env_camera')
    #     seg = camera.render(segmentation=True)
    #     # Display the contents of the first channel, which contains object
    #     # IDs. The second channel, seg[:, :, 1], contains object types.
    #     geom_ids = seg[:, :, 0]
    #     # clip to bool variables
    #     pixels = geom_ids.clip(min=0, max=1)  # shape (height, width)
    #     # draw
    #     if show:
    #         fig, ax = plt.subplots(1, 1)
    #         ax.imshow(1-pixels, cmap='gray')
    #     # # move the robot and the goal back
    #     # with physics.reset_context():
    #     #     self._gripper1.set_pose(physics, position=origin_gripper_pos)
    #     #     physics.data.qpos[0:7] = origin_qpos[0:7]
    #     #     physics.data.qpos[7:-3] = origin_qpos[7:-3]
    #     #     physics.data.qpos[-3:] = origin_qpos[-3:]
    #     #     physics.data.qvel[0:7] = origin_qvel[0:7]
    #     #     physics.data.qvel[7:-3] = origin_qvel[7:-3]
    #     #     physics.data.qvel[-3:] = origin_qvel[-3:]
    #     #     # self._rope.set_pose(physics, position=origin_rope_pos[0:3], quaternion=origin_rope_pos[3:])
    #     #     # self._rope.set_velocity(physics, velocity=origin_rope_vel[0:3], angular_velocity=origin_rope_vel[3:])
    #     #
    #     #     # self._gripper1.set_velocity(physics, velocity=origin_gripper_vel)
    #     #     # physics.bind(self._gripper1.mjcf_model.find_all('joint')).qpos = origin_gripper_pos
    #     #     # physics.bind(self._gripper1.mjcf_model.find_all('joint')).qvel = origin_gripper_vel
    #     return pixels


if __name__ == "__main__":
    task = RopeManipulation()
    seed = None
    env = composer.Environment(task, random_state=seed)
    obs = env.reset()

    def dummy_controller(timestep):
        print(timestep.observation['pos_0'])
        print("------------------")
        return 0


    controller = RandomController(udim=3, urange=1, horizon=40, sigma=20, lower_bound=[-1, -1, -1], upper_bound=[1, 1, 1])
    action_seq = controller.step(None)
    i = 0
    image = None
    def random_policy(time_step):
        # time.sleep(0.1)
        global i, action_seq, image
        if time_step.step_type == StepType.FIRST:
            # env._random_state.seed(0)
            # print(time_step.observation)
            action_seq = controller.step(x=None)
            i = 0
            # robot_pos = env.physics.bind(env._task.rob_freejoint).qpos
            # robot_vel = env.physics.bind(env._task.rob_freejoint).qvel
            # robot_pos[0:2] = [0.1, 0.1]
            # robot_vel[0:2] = [0, 0]
        # print(time_step.reward)

        if i < len(action_seq):
            action = action_seq[i]
            i += 1
        else:
            action = 0
            # action_seq = controller.step(x=None)
            # i = 0
        # print(f"{env.physics.data.time}s: {action}")

        # print("Real goal pos:", env._task._goal_indicator.pos[0:2], "Desired goal pose:", env._task.goal[0:2])
        # print(action)
        if i == 20:
            image = env.task.get_aerial_view(env.physics, show=False)
        print(action)
        return action

    viewer.launch(env, policy=random_policy)
    # ipdb.set_trace()
