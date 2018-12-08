import gym
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from torch.distributions import Normal
import numpy as np


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 1.0)


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
            nn.Linear(input_size, hidden_size),
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


def main():
    # Hyper parameters
    epochs = 30
    env_samples = 100
    episode_length = 1000
    ppo_epochs = 4
    batch_size = 256
    epsilon = 0.2
    lr = 1e-3
    env = gym.make('Pendulum-v0')
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], hidden_size=32)

    val_loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    val_losses = []
    policy_losses = []

    loop = tqdm(total=(epochs), position=0)
    episode_avg_rewards = []

    for e in range(epochs):

        experience = []
        loop.update(1)
        rewards = []

        # Create env_samples number of episode rollouts
        for j in range(env_samples):

            state = env.reset()
            state = state
            rollout = []

            # Each action in an episode
            for k in range(episode_length):
                torch_state = torch.FloatTensor(state).unsqueeze(0)
                dist, val = model(torch_state)

                action = dist.sample().numpy()[0]

                obs, reward, terminal, _ = env.step(action)
                rewards.append(reward)

                log_prob = dist.log_prob(torch.tensor(action))

                rollout.append((state, reward, log_prob.detach().numpy()[0], action))
                state = obs

                if j is -1:
                    env.render()

                if terminal:
                    break

            compute_returns(rollout)
            experience.append(rollout)

        avg_rewards = sum(rewards) / env_samples
        episode_avg_rewards.append(avg_rewards)
        print(" ")
        print("Epoch: ", e, " Avg Reward: ", avg_rewards)

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


if __name__ == '__main__':
    main()
