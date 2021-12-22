# Reference: https://arxiv.org/pdf/1509.02971.pdf
# https://github.com/seungeunrho/minimalRL/blob/master/ddpg.py
# https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ddpg/ddpg.py

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from rl_agents import buffers, networks, noise

def compute_t_r(exp):
    d = np.linalg.norm(agent_obs_to_goal(exp['s1']) - exp['g'])
    r = 0 if d > 0.1 else 1
    return False, r

def exp_to_tuple(exp):
    return (
        np.concatenate([exp['s0'], exp['g']]),
        exp['a'],
        exp['r'],
        np.concatenate([exp['s1'], exp['g']]),
        exp['t']
    )

def sample_goal():
    return np.random.uniform(
        low=np.array([-0.27, -0.27]),
        high=np.array([0.27,0.27])
    )

def env_obs_to_agent_obs(obs):
    return np.concatenate([obs[:2], obs[6:]])

def agent_obs_to_goal(obs):
    return obs[:2]

def run(
        env,
        args,
        writer
):

    device = torch.device('cuda' if torch.cuda.is_available() and args['--cuda'] else 'cpu')

    # ALGO LOGIC: initialize agent here:
    max_action = float(env.action_space.high[0])
    rb = buffers.ReplayBuffer(args['--buffer_size'])
    actor = networks.Actor(input_size=8, output_size=2).to(device)
    n_actions = env.action_space.shape[-1]
    action_noise = noise.OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))
    qf1 = networks.QNetwork(input_size=10).to(device)
    qf1_target = networks.QNetwork(input_size=10).to(device)
    target_actor = networks.Actor(input_size=8, output_size=2).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args['--learning_rate'])
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args['--learning_rate'])
    loss_fn = nn.MSELoss()

    obs = env.reset()
    episode_reward, episode_steps = 0, 0
    episode = []
    current_exp = {}
    current_exp['g'] = sample_goal()

    for global_step in range(args['--total_timesteps']):

        current_exp['s0'] = env_obs_to_agent_obs(obs)

        # Action logic here
        if global_step < args['--learning_starts']:
            action = action_noise()
        else:
            state = np.concatenate([current_exp['s0'], current_exp['g']])
            action = actor.forward(state.reshape((1,)+state.shape), device)
            action = (
                    action.tolist()[0]
                    + np.random.normal(0, max_action * args['--exploration_noise'], size=env.action_space.shape[0])
            ).clip(env.action_space.low, env.action_space.high)
        current_exp['a'] = action

        # Step
        next_obs, _, _, _ = env.step(action)
        current_exp['s1'] =  env_obs_to_agent_obs(next_obs)
        current_exp['t'], current_exp['r'] = compute_t_r(current_exp)
        episode_reward += current_exp['r']
        episode_steps += 1
        episode.append(current_exp.copy())

        # ALGO LOGIC: training.
        if global_step > args['--learning_starts']:

            for _ in range(4):

                s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args['--batch_size'])

                with torch.no_grad():

                    next_state_actions = (
                        target_actor.forward(s_next_obses, device)
                    ).clamp(env.action_space.low[0], env.action_space.high[0])
                    # qf1_next_target = qf1_target.forward(s_next_obses, next_state_actions, device)
                    qf1_next_target = qf1_target.forward(s_next_obses, next_state_actions, device).clamp(
                        0 / (1 - args['--gamma']),
                        1 / (1 - args['--gamma'])
                    )
                    next_q_value = torch.Tensor(s_rewards).to(device) + (1 - torch.Tensor(s_dones).to(device)) * args['--gamma'] * (qf1_next_target).view(-1)

                qf1_a_values = qf1.forward(s_obs, torch.Tensor(s_actions).to(device), device).view(-1)
                qf1_loss = loss_fn(qf1_a_values, next_q_value)

                q_optimizer.zero_grad()
                qf1_loss.backward()
                # nn.utils.clip_grad_norm_(list(qf1.parameters()), args['--max_grad_norm'])
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

                    # nn.utils.clip_grad_norm_(list(actor.parameters()), args['--max_grad_norm'])
                    actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args['--tau'] * param.data + (1 - args['--tau']) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args['--tau'] * param.data + (1 - args['--tau']) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)

        # CRUCIAL step easy to overlook
        obs = next_obs

        if current_exp['t'] or episode_steps >= args['--episode_length']:

            # Record rewards for plotting purposes
            print(f"global_step={global_step}, episode_reward={episode_reward}")
            writer.add_scalar("charts/episodic_return", episode_reward, global_step)
            writer.add_scalar("charts/success", int(episode_reward > 0), global_step)

            # HER
            virtual_goal_idxs = np.random.choice(range(100), 4)
            for i, exp in enumerate(episode):
                rb.put(exp_to_tuple(exp))
                for idx in virtual_goal_idxs:
                    if i <= idx:
                        v_exp = exp.copy()
                        v_exp['g'] = agent_obs_to_goal(episode[idx]['s1'])
                        v_exp['t'], v_exp['r'] = compute_t_r(v_exp)
                        rb.put(exp_to_tuple(v_exp))

            obs = env.reset()
            episode_reward, episode_steps = 0, 0
            episode = []
            current_exp = {}
            current_exp['g'] = sample_goal()

    env.close()
    writer.close()
