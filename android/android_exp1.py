import gym
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from torch.distributions import Normal
import numpy as np
import holodeck
from holodeck.sensors import *
import sys


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.2)


# Define the Actor Critic
class ActorCritic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=10, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size*2),
            nn.ReLU(),
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        self.log_std = nn.Parameter(torch.ones(1, output_size) * std)
        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value


class ExperienceDataset(Dataset):
    def __init__(self, experience):
        super(ExperienceDataset, self).__init__()
        self._exp = []
        for x in experience:
            self._exp.extend(x)
        self._length = len(self._exp)

    def __getitem__(self, index):
        return self._exp[index]

    def __len__(self):
        return self._length


def compute_returns(rollout, gamma=0.9):
    ret = 0

    for i in reversed(range(len(rollout))):
        obs, reward, action_dist, action = rollout[i]
        ret = reward + gamma * ret
        rollout[i] = (obs, reward, action_dist, action, ret)


def getObsVector(state, joints):
    state = np.concatenate((state[Sensors.LOCATION_SENSOR],
                            state[Sensors.ORIENTATION_SENSOR],
                            state[Sensors.RELATIVE_SKELETAL_POSITION_SENSOR],
                            state[Sensors.JOINT_ROTATION_SENSOR][0:joints]),
                           axis=None)
    return state


def AndroidTest(exp_name="exp", lr=1e-4, env_samples=20, epochs=10,
                episode_length=500, gamma=0.99,
                start_steps=100, energy_cost_weight=0.0,
                reward_type="x_dist_max", hidden_size=256, print_to_file=True):

    print("Beginning test ", exp_name)

    # Hyper parameters
    ppo_epochs = 4
    batch_size = 512
    epsilon = 0.2
    joints = 54
    render = False
    action_multiplier = 3

    env = holodeck.make('ExampleLevel', window_res=[256, 256])
    raw_state, rew, done, _ = env.reset()
    state = getObsVector(raw_state, joints)

    input_len = state.shape[0]

    model = ActorCritic(input_len, joints, hidden_size=hidden_size)

    val_loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    val_losses = []
    policy_losses = []

    episode_avg_rewards = []

    for e in range(epochs):

        experience = []
        rewards = []

        # Create env_samples number of episode rollouts
        for j in range(env_samples):

            raw_state, rew, done, _ = env.reset()

            for _ in range(start_steps):
                env.tick()

            raw_state, rew, done, _ = env.step(np.zeros(94))

            max_dist = raw_state[Sensors.LOCATION_SENSOR][0]

            state = getObsVector(raw_state, joints)
            rollout = []

            # Each action in an episode
            for k in range(episode_length):
                torch_state = torch.FloatTensor(state).unsqueeze(0)
                dist, val = model(torch_state)

                action = dist.sample().numpy()[0]

                obs_raw, reward, terminal, _ = env.step(action_multiplier*np.append(action, np.zeros((94-54))))
                distance = obs_raw[Sensors.LOCATION_SENSOR][0]

                energy_cost = np.mean(np.abs(action)) * energy_cost_weight

                if reward_type is "z_dist":
                    reward = obs_raw[Sensors.LOCATION_SENSOR][2]
                elif reward_type is "x_dist":
                    reward = obs_raw[Sensors.LOCATION_SENSOR][0]
                else:
                    if distance > max_dist:
                        reward = distance-max_dist
                        max_dist = distance
                    else:
                        reward = 0

                reward -= energy_cost
                reward *= 10

                obs = getObsVector(obs_raw, joints)
                rewards.append(reward)

                log_prob = dist.log_prob(torch.tensor(action))

                rollout.append((state, reward, log_prob.detach().numpy()[0], action))
                state = obs

                if j is -1:
                    env.render()

                if terminal:
                    break

            compute_returns(rollout, gamma=gamma)
            experience.append(rollout)

        avg_rewards = sum(rewards) / env_samples
        episode_avg_rewards.append(avg_rewards)
        print(" ")
        print("Epoch: ", e, "/", epochs, " Avg Reward: ", avg_rewards)

        exp_data = ExperienceDataset(experience)
        exp_loader = DataLoader(exp_data, batch_size=batch_size, shuffle=True, pin_memory=True)

        for _ in range(ppo_epochs):
            # Train network on batches of states
            for observation, reward, old_log_prob, action, ret in exp_loader:
                optimizer.zero_grad()
                new_dist, value = model(observation.float())

                ret = ret.unsqueeze(1)

                advantage = ret.float() - value.detach()

                new_log_prob = new_dist.log_prob(action)

                r_theta = (new_log_prob - old_log_prob).exp()

                clipped = r_theta.clamp(1 - epsilon, 1 + epsilon)

                objective = torch.min(r_theta * advantage, clipped * advantage)

                policy_loss = -torch.mean(objective)
                val_loss = val_loss_func(ret.float(), value)

                loss = policy_loss + val_loss
                loss.backward()

                optimizer.step()
                val_losses.append(val_loss.detach().numpy())
                policy_losses.append(policy_loss.detach().numpy())

        if e % 10 == 0:
            model_name = 'Dec10Exp/Exp' + exp_name + str(e) + '_reward_' + str(int(avg_rewards)) + '.model'
            torch.save(model, model_name)


if __name__ == '__main__':
    exp = int(sys.argv[1])

    if exp is 0:
        AndroidTest(exp_name="0-Control")
    elif exp is 1:
        AndroidTest(exp_name="1-Stand", reward_type="z_dist")
    elif exp is 2:
        AndroidTest(exp_name="2-HigherLR", lr=5e-4)
    elif exp is 3:
        AndroidTest(exp_name="3-LowerLR", lr=5e-5)
    elif exp is 4:
        AndroidTest(exp_name="4-MoreSamples", env_samples=50)
    elif exp is 5:
        AndroidTest(exp_name="5-LargerHiddenSize", hidden_size=512)
    elif exp is 6:
        AndroidTest(exp_name="6-SmallerHiddenSize", hidden_size=128)
    elif exp is 7:
        AndroidTest(exp_name="7-SmallEnergyCost", energy_cost_weight=1e-8)
    elif exp is 8:
        AndroidTest(exp_name="8-MedEnergyCost", energy_cost_weight=1e-6)
    elif exp is 9:
        AndroidTest(exp_name="9-StartStepsZero", start_steps=0)
    elif exp is 10:
        AndroidTest(exp_name="10-LowerGamma", gamma=0.9)
    elif exp is 11:
        AndroidTest(exp_name="11-JustX_Dist", reward_type="x_dist")