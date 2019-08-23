import sys

#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import os
import numpy as np

from PIL import Image

import robosuite
from robosuite.wrappers import TeleopWrapper, GymWrapper, IKWrapper

from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, CnnLstmPolicy, CnnLnLstmPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import PPO2, TRPO, DDPG
from stable_baselines.bench import Monitor

best_mean_reward, n_steps = -np.inf, 0

name = 'cooked_trpo_vannilla_cnn_teleop_wrapper'
log_dir = "./learning/checkpoints/lift/" + name + '/'
os.makedirs(log_dir, exist_ok=True)

def callback(_locals, _globals):
    global n_steps, best_mean_reward
    if (n_steps + 1) % 5 == 0:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        if len(x) > 0:
            mean_reward = np.mean(y[-20:])
            print(x[-1], 'timesteps')
            print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))
            print("Saving running normalization avg")
            env.save_running_average(log_dir)
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                print("Saving new best model")
                _locals['self'].save(log_dir + 'best_model.pkl')
    n_steps += 1
    return True

def main():
    num_stack = 5
    num_env = 1
    image_state = True
    subproc = False
    markov_obs = False
    finger_obs = False
    env_type = "SawyerLift" # "SawyerReach"
    arch = CnnPolicy
    render = False

    #existing = '/Users/aqua/Documents/workspace/summer/svl_summer/robosuite/robosuite/learning/checkpoints/lift/vannilla_cnn_teleop_wrapper/best_model.pkl'
    existing = None
    if existing:
        render = True
        num_env = 1
        subproc = False
    print('Config for ' + log_dir + ':')
    print('num_stack:', num_stack)
    print('num_env:', num_env)
    print('render:', render)
    print('image_state:', image_state)
    print('subproc:', subproc)
    print('existing:', existing)
    print('markov_obs:', markov_obs)
    print('log_dir:', log_dir)

    global env
    env = []
    for i in range(num_env):
        ith = GymWrapper(IKWrapper(robosuite.make(env_type, has_renderer=render, has_offscreen_renderer=image_state, use_camera_obs=image_state, reward_shaping=True, camera_name='agentview', camera_height=64, camera_width=64)), num_stack=num_stack, keys=['image'])
        ith.metadata = {'render.modes': ['human']}
        ith.reward_range = None
        ith.spec = None
        ith = Monitor(ith, log_dir, allow_early_resets=True)
        env.append((lambda: ith))

    # TODO: Set normalization values for TRPO
    if num_stack:
        env = VecFrameStack(VecNormalize(SubprocVecEnv(env, 'fork'), norm_obs=True, norm_reward=False, clip_obs=1e10, clip_reward=1e10), num_stack) if subproc else VecFrameStack(VecNormalize(DummyVecEnv(env), norm_obs=True, norm_reward=False, clip_obs=1e10, clip_reward=1e10), num_stack)
        env.load_running_average(log_dir)
    else:
        env = SubprocVecEnv(env, 'fork') if subproc else DummyVecEnv(env)

    if existing:
        print('Loading pkl directly')
        model = TRPO.load(existing)
    else:
        try:
            print('Trying existing model...')
            model = TRPO.load(log_dir + 'best_model.pkl')
            model.set_env(env)
        except:
            print('No existing model found. Training new one.')
            #n_actions = env.action_space.shape[-1]
            #param_noise = None
            #action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
            #model = DDPG('CnnPolicy', env, verbose=1, param_noise=param_noise, action_noise=action_noise)
            model = TRPO(arch, env, verbose=1, timesteps_per_batch=1024)
            #model = PPO2(arch, env, verbose=1, nminibatches=num_env)

        model.learn(total_timesteps=int(1e8), callback=callback)

    obs = env.reset()
    while True:
        #obs = np.tile(obs, (8, 1))
        obsn = obs[0][:, :, 6:9]
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        if render:
            env._get_target_envs([0])[0].render()
        if done[0]:
            obs = env.reset()

if __name__ == '__main__':
    main()

