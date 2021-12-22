# Reference: https://arxiv.org/pdf/1509.02971.pdf
# https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
# https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from gym.wrappers import Monitor
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
from rl_agents import buffers, networks, noise, utils

def run(
        env,
        args,
        writer
):

    device = torch.device('cuda' if torch.cuda.is_available() and args['--cuda'] else 'cpu')

    assert isinstance(env.action_space, Box), "only continuous action space is supported"

    # ALGO LOGIC: initialize agent here:
    max_action = float(env.action_space.high[0])
    rb = buffers.ReplayBuffer(args['--buffer_size'])
    actor = networks.Actor(env).to(device)
    n_actions = env.action_space.shape[-1]
    action_noise = noise.OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
    qf1 = networks.QNetwork(env).to(device)
    qf1_target = networks.QNetwork(env).to(device)
    target_actor = networks.Actor(env).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args['--learning_rate'])
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args['--learning_rate'])
    loss_fn = nn.MSELoss()
    # TRY NOT TO MODIFY: start the game
    obs = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode = []

    for global_step in range(args['--total_timesteps']):
        # ALGO LOGIC: put action logic here

        if global_step < args['--learning_starts']:
            action = action_noise()
        else:
            action = actor.forward(obs.reshape((1,)+obs.shape), device)
            action = (
                    action.tolist()[0]
                    + np.random.normal(0, max_action * args['--exploration_noise'], size=env.action_space.shape[0])
            ).clip(env.action_space.low, env.action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, info = env.step(action)
        episode_reward += reward
        episode_steps += 1
        # episode.append(exp.copy())

        # ALGO LOGIC: training.
        rb.put((obs, action, reward, next_obs, done))

        if global_step >= args['--learning_starts']:

            s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args['--batch_size'])

            with torch.no_grad():

                next_state_actions = (
                    target_actor.forward(s_next_obses, device)
                ).clamp(env.action_space.low[0], env.action_space.high[0])
                qf1_next_target = qf1_target.forward(s_next_obses, next_state_actions, device)
                # qf1_next_target = qf1_target.forward(s_next_obses, next_state_actions, device).clamp(
                #     -0.01 / (1 - args['--gamma']),
                #     0.99
                # )
                next_q_value = torch.Tensor(s_rewards).to(device) + (1 - torch.Tensor(s_dones).to(device)) * args['--gamma'] * (qf1_next_target).view(-1)

            qf1_a_values = qf1.forward(s_obs, torch.Tensor(s_actions).to(device), device).view(-1)
            qf1_loss = loss_fn(qf1_a_values, next_q_value)

            # optimize the midel
            q_optimizer.zero_grad()
            qf1_loss.backward()
            nn.utils.clip_grad_norm_(list(qf1.parameters()), args['--max_grad_norm'])
            q_optimizer.step()

            if global_step % args['--policy_frequency'] == 0:

                actor_optimizer.zero_grad()
                a = actor.forward(s_obs, device)
                actor_loss = -qf1.forward(s_obs, a, device).mean()

                # actor_loss.backward()
                # Here with inverted gradients
                actor_loss.backward(inputs=a, retain_graph=True)
                l = torch.Tensor(env.action_space.low)
                h = torch.Tensor(env.action_space.high)
                w = h - l
                grad_pos = torch.gt(a.grad, 0)
                grad_inverter = torch.where(grad_pos, (a - l) / w, (h - a) / w)
                inverted_grads = torch.mul(a.grad, grad_inverter)
                a.backward(inverted_grads, inputs=list(actor.parameters()))

                nn.utils.clip_grad_norm_(list(actor.parameters()), args['--max_grad_norm'])
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args['--tau'] * param.data + (1 - args['--tau']) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args['--tau'] * param.data + (1 - args['--tau']) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        if done or episode_steps >= args['--episode_length']:
            # TRY NOT TO MODIFY: record rewards for plotting purposes
            print(f"global_step={global_step}, episode_reward={episode_reward}")
            writer.add_scalar("charts/episodic_return", episode_reward, global_step)
            obs, episode_reward = env.reset(), 0
            episode_steps = 0

    env.close()
    writer.close()
