# HW12: the gym 0.18 API the 2021 sample was written for, on top of Gymnasium 1.4.
# Not part of the sample. Every difference from gym 0.18 is listed here:
# - gym.make('LunarLander-v2') builds Gymnasium's LunarLander-v3 (v2 is no longer registered).
# - env.seed(s) seeds the next reset(); later reset() calls continue the same RNG, as in gym 0.18.
#   The RNG itself differs (Gymnasium's np.random.Generator), so terrains are not those of 2021.
# - reset() returns obs only; step() returns (obs, reward, done, info) with done = terminated or truncated.
# - render_mode stays None: Gymnasium only creates the engine particles when rendering,
#   so the sample's render() calls are dropped instead of supported.
import gymnasium

RENAMED = {'LunarLander-v2': 'LunarLander-v3'}


class OldAPIEnv:
    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._next_seed = None

    def seed(self, seed=None):
        self._next_seed = seed
        return [seed]

    def reset(self):
        obs, _ = self.env.reset(seed=self._next_seed)
        self._next_seed = None
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated or truncated, info


def make(name):
    return OldAPIEnv(gymnasium.make(RENAMED.get(name, name)))
