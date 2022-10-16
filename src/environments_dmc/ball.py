from dm_control import composer
from dm_control import mjcf
from dm_control import viewer
from dm_control.composer.observation import observable  # For exposing observations
from dm_control.composer import variation   # Implements model variations
from dm_control.composer import Arena   # Blank canvas arena
import numpy as np


NUM_SUBSTEPS = 25  # The number of physics sub-steps per control time step.


class Ball(composer.Entity):
    """A Ball derived from composer.Entity base class"""
    def _build(self, radius, mass):
        # Create a mjcf root associated with this entity
        self._mjcf_model = mjcf.RootElement()
        # Pick a random opaque color for the ball
        rgba = np.random.uniform([0, 0, 0, 1], [1, 1, 1, 1])
        # Since this entity is a single object geometry, we can add that sphere geometry and assign as _geom
        self._geom = self._mjcf_model.worldbody.add('geom', name='sphere', type='sphere', size=[radius], rgba=rgba)

    def _build_observables(self):
        return BallObservables(self)

    @property
    def mjcf_model(self):
        return self._mjcf_model

    # Unused actuators property for balls
    @property
    def actuators(self):
        return tuple(self._mjcf_model.find_all('actuator'))


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


# The task the ball has to complete is to fall under gravity and fall out of view
# A composer.Task object combines all the Entities (just Ball here) and is used with a composer.Environment Wrapper
class FallOutOfView(composer.Task):
    # Set up the scene
    def __init__(self, ball):
        # Create an Arena for the task using default blank Arena
        self._arena = Arena()
        # Add the ball object to the Arena
        self._ball = ball
        self._arena.add_free_entity(self._ball)
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
        self._ball.observables.ball_position.enabled = True
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
    ball = Ball(0.1, 1.0)

    task = FallOutOfView(ball)
    seed = None
    env = composer.Environment(task, random_state=seed)
    obs = env.reset()

    viewer.launch(env)

