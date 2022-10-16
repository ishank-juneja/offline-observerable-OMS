## New rope attempt with no floor in arena

from dm_control import composer
from dm_control import mjcf
from dm_control import viewer
from dm_control.composer.observation import observable  # For exposing observations
from dm_control.composer import variation   # Implements model variations
from dm_control.composer import Arena   # Blank canvas arena
import numpy as np


NUM_SUBSTEPS = 25  # The number of physics sub-steps per control time step.


class RopeEntity(composer.Entity):
    def _build(self, length=20, rgba=(0.2, 0.8, 0.2, 1)):
        self._model = mjcf.RootElement('rope')
        self._model.compiler.angle = 'radian'
        # self._model.default.geom.friction = [1, 0.1, 0.1]
        body = self._model.worldbody.add('body', name='rB0')
        self._composite = body.add('composite', prefix="r", type='rope', count=[length, 1, 1], spacing=0.084)
        self._composite.add('joint', kind='main', damping=0.01, stiffness=0.01)
        self._composite.geom.set_attributes(type='capsule', size=[0.0375, 0.04], rgba=rgba, mass=0.005,
                                            contype=1, conaffinity=1, priority=1, friction=[0.1, 5e-3, 1e-4])
        self.reset_friction()

    @property
    def mjcf_model(self):
        return self._model

    def reset_friction(self, random_state=None):
        pass

    def reset_pose(self):
        pass


class CupEntity(composer.Entity):
    def _build(self):
        self._model = mjcf.RootElement('cup')
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


class Ball(composer.Entity):
    """A Ball derived from composer.Entity base class"""
    def _build(self, radius, mass):
        # Create a mjcf root associated with this entity
        self._model = mjcf.RootElement('ball')
        body = self._model.worldbody.add('body', name='dummy')
        # Pick a random opaque color for the ball
        # rgba = np.random.uniform([0, 0, 0, 1], [1, 1, 1, 1])
        rgba = [1, 0, 0, 1]
        # Since this entity is a single object geometry, we can add that sphere geometry and assign as _geom
        self._geom = body.add('geom', name='sphere', type='sphere', size=[radius], rgba=rgba)

    def _build_observables(self):
        return BallObservables(self)

    @property
    def mjcf_model(self):
        return self._model

    # Unused actuators property for balls
    @property
    def actuators(self):
        return tuple(self._model.find_all('actuator'))


class BallObservables(composer.Observables):
    """Observable quantities for the ball/freely falling sphere."""
    @composer.observable
    def ball_position(self):
        # Return ptrs to all geoms, just a sphere in case of ball
        sphere_geom = self._entity.mjcf_model.find_all('geom')
        return observable.MJCFFeature('pos', sphere_geom)

    @composer.observable
    def ball_velocity(self):
        # Return ptrs to all geoms, just a sphere in case of ball
        sphere_geom = self._entity.mjcf_model.find_all('geom')
        return observable.MJCFFeature('vel', sphere_geom)


# A dummy Kendama task with no goal in particular
class Kendama(composer.Task):
    # Set up the scene
    def __init__(self, rope_length=11, action_noise: float = 0.0, friction_noise: float = 0.2):
        # Create an Arena for the task using default blank Arena
        self._arena = Arena()

        # Simulation related params settings
        self._arena.mjcf_model.compiler.inertiafromgeom = True
        self._arena.mjcf_model.default.joint.damping = 0
        self._arena.mjcf_model.default.joint.stiffness = 0
        self._arena.mjcf_model.default.geom.contype = 3
        self._arena.mjcf_model.default.geom.conaffinity = 3
        self._arena.mjcf_model.default.geom.friction = [1, 0.1, 0.1]
        self._arena.mjcf_model.option.gravity = [1e-5, 0, -9.81]
        self._arena.mjcf_model.option.integrator = 'Euler'  # RK4 or Euler
        self._arena.mjcf_model.option.timestep = 0.001

        # The rope of the cup and ball game
        self._rope = RopeEntity(length=rope_length)
        # The cup in the cup and ball game
        self._cup = CupEntity()
        # The ball in the cup and ball game
        # self._ball = Ball(0.1, 1.0)

        rope_site = self._arena.add_free_entity(self._rope)
        cup_site = self._arena.attach(self._cup)
        # ball_site = self._arena.attach(self._ball)

        rope_site.pos = [0.0, 0.0, 1.0]
        cup_site.pos = [0.0, 0.0, 1.0]
        # ball_site.pos = [1.0, 0.0, 0.0]

        # constraint
        self._arena.mjcf_model.equality.add('connect', body1='cup/dummy', body2='rope/rB0', anchor=[0.0, 0, 0.0])
        # self._arena.mjcf_model.equality.add('connect', body2='rope/rB0', body1='ball/dummy', anchor=[0.0, 0.0, 0.0])

        # Configure lighting
        self._arena.mjcf_model.worldbody.add('light', pos=(0, 0, 4))

        # Configure initial pose
        self._ball_initial_pose = (0., 0., 0.)

        # Configure variators
        # For varying MJCF attributes like geom size
        # Called before initializing an episode before the model is compiled
        self._mjcf_variator = variation.MJCFVariator()
        # Changes bound attributes like external forces
        # Called after model has been compiled
        self._physics_variator = variation.PhysicsVariator()

        # Enable observables
        # self._ball.observables.ball_position.enabled = True
        # self._ball.observables.ball_velocity.enabled = True

        # NUM_SUBSTEPS physics steps per control time step
        self.control_timestep = NUM_SUBSTEPS * self.physics_timestep
        self._task_observables = {}

    #  - - - - call_backs - - - - -

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        self._physics_variator.apply_variations(physics, random_state)

    def get_reward(self, physics):
        return 0.0


if __name__ == '__main__':
    task = Kendama()
    seed = None
    env = composer.Environment(task, random_state=seed)
    obs = env.reset()

    viewer.launch(env)

