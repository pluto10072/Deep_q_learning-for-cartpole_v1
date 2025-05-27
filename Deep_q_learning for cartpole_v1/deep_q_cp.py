import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt  # 新增

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建CartPole环境
env = gym.make('CartPole-v1')
num_states = env.observation_space.shape[0]  # 4
num_actions = env.action_space.n  # 2


# 定义DQN网络结构
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)


learning_rate = 0.001
gamma = 0.9
epsilon_start = 0.5
epsilon_end = 0.001
epsilon_decay = 0.995
episodes = 200
batch_size = 32
buffer_size = 1000000

policy_net = DQN(num_states, num_actions).to(device)
target_net = DQN(num_states, num_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
memory = deque(maxlen=buffer_size)


def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        with torch.no_grad():
            state_arr = np.array(state, dtype=np.float32)
            state_tensor = torch.from_numpy(state_arr).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            return q_values.argmax().item()


def store_transition(s, a, r, s_, d):
    memory.append((s, a, r, s_, d))


def update():
    if len(memory) < batch_size:
        return
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q = target_net(next_states).max(1)[0]
    expected_q = rewards + (1 - dones) * gamma * next_q

    loss = nn.MSELoss()(current_q, expected_q.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # scheduler.step() # 学习率衰减


episode_rewards = []  # 保存每个回合的总奖励
epsilon = epsilon_start
best_reward = float('-inf')
best_model_state = None
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = choose_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        position = state[0]
        angle = state[2]
        # reward -= 1.5*abs(position)
        # reward -= 1.5 * abs(angle)
        done = terminated or truncated
        store_transition(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        update()

    episode_rewards.append(total_reward)  # 保存每回合reward

    # 保存表现最好的网络参数
    if total_reward > best_reward:
        best_reward = total_reward
        best_model_state = policy_net.state_dict()

    if episode % 20 == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # epsilon = max(epsilon_end, epsilon * epsilon_decay)
    epsilon -= epsilon_start/episodes

    if episode % 100 == 0:
        print(f"Episode {episode} 已完成")
    # print(f"Episode {episode + 1}, Total Reward: {total_reward}")


# 绘制训练reward曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, episodes + 1), episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('DQN Training Reward Curve on CartPole-v1')
plt.grid(True)
plt.show()

# 训练结束后加载表现最好的网络参数
if best_model_state is not None:
    policy_net.load_state_dict(best_model_state)
    print(f"已加载reward最高({best_reward})的策略参数用于测试。")


# 用最优策略测试10个episode
test_rewards = []
for ep in range(10):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        state_arr = np.array(state, dtype=np.float32)
        state_tensor = torch.from_numpy(state_arr).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
    test_rewards.append(total_reward)
    print(f"Test Episode {ep+1}: Total Reward = {total_reward}")

print(f"10次测试平均reward: {np.mean(test_rewards)}")
"""
# 测试并录制视频
video_env = RecordVideo(
    gym.make('CartPole-v1', render_mode="rgb_array"),
    video_folder="./cartpole_test_video",
    name_prefix="dqn_test"
)

state, _ = video_env.reset()
done = False
steps = 0
while not done and steps < 500:
    state_arr = np.array(state, dtype=np.float32)
    state_tensor = torch.from_numpy(state_arr).unsqueeze(0).to(device)
    q_values = policy_net(state_tensor)
    action = q_values.argmax().item()
    next_state, reward, terminated, truncated, _ = video_env.step(action)
    done = terminated or truncated
    # print(f"Step {steps}: State={state}, Action={action}, Next State={next_state}")
    state = next_state
    steps += 1

video_env.close()  # 确保视频写入

"""

env.close()
