import abc
import collections
import glob
import os
import random
import time
from typing import List, Tuple
import os
from gymnasium.wrappers import RecordVideo
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch._tensor import Tensor
from torch.utils.tensorboard import SummaryWriter


class BaseModel(nn.Module):
    def __init__(self, num_inputs=4, num_outputs=2):
        super(BaseModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class DuelingModel(nn.Module):
    def __init__(self, num_inputs=4, num_outputs=2):
        super(DuelingModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_inputs, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )
        self.Q_net = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_outputs),
        )
        self.V_net = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        v_val = self.V_net(x)
        q_val = self.Q_net(x)
        return v_val, q_val


class Data:
    def __init__(self, state, action, reward, next_state, episode, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.episode = episode
        self.done = done


class Memory:
    def __init__(self, capacity: int):
        self.buffer: collections.deque[Data] = collections.deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def size(self):
        return len(self.buffer)

    def set(self, data: Data):
        # TODO
        self.buffer.append(data)

    def get(self, batch_size: int) -> List[Data]:
        # TODO
        assert batch_size < len(self.buffer), "no enough samples to sample from"
        return random.sample(self.buffer, batch_size)

    def remove_last(self):
        """删除最后加入的 episode 的数据"""
        episode_last = self.buffer[-1].episode
        while self.buffer[-1].episode == episode_last and len(self.buffer) >= MIN_CAPACITY:
            self.buffer.pop()


def samples_to_tensors(samples: List[Data]):
    num_samples = len(samples)

    states_shape = (num_samples,) + samples[0].state.shape
    states = np.zeros(states_shape, dtype=np.float32)
    next_states = np.zeros(states_shape, dtype=np.float32)

    rewards = np.zeros(num_samples, dtype=np.float32)
    actions = np.zeros(num_samples, dtype=np.int64)
    non_ends = np.zeros(num_samples, dtype=np.float32)

    for i, s in enumerate(samples):
        states[i] = s.state
        next_states[i] = s.next_state
        rewards[i] = s.reward
        actions[i] = s.action
        non_ends[i] = 0.0 if s.done else 1.0

    states = torch.from_numpy(states).cuda()
    actions = torch.from_numpy(actions).cuda()
    rewards = torch.from_numpy(rewards).cuda()
    next_states = torch.from_numpy(next_states).cuda()
    non_ends = torch.from_numpy(non_ends).cuda()

    return states, actions, rewards, next_states, non_ends


class BaseDQN(metaclass=abc.ABCMeta):
    def __init__(self):
        super(BaseDQN, self).__init__()
        self.learn_step_counter = 0
        self.eval_net = None
        self.target_net = None
        self.optimizer = None
        self.loss_func = None

    def choose_action(self, state, EPSILON=1.0):
        state = torch.tensor(state, dtype=torch.float).to(device)
        if np.random.random() > EPSILON:  # random number
            # greedy policy
            with torch.no_grad():
                action_value = self.eval_net.forward(state)
            action = torch.argmax(action_value).item()
        else:
            # random policy
            action = np.random.randint(0, NUM_ACTIONS)  # int random number
        return action

    def model_step(self):
        self.learn_step_counter += 1
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.learn_step_counter % SAVING_IETRATION == 0:
            self.save_train_model(self.learn_step_counter)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        loss = self.loss_func(preds, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    @abc.abstractmethod
    def evaluating(self, samples: List[Data]) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def learn(self, samples):
        # update the parameters
        self.model_step()
        preds, targets = self.evaluating(samples)
        self.update(preds, targets)

    def save_train_model(self, epoch):
        torch.save(self.eval_net.state_dict(), f"{SAVE_PATH_PREFIX}/ckpt/{epoch}.pth")

    def load_net(self, file):
        self.eval_net.load_state_dict(torch.load(file))
        self.target_net.load_state_dict(torch.load(file))

    def name(self):
        return self.__class__.__name__


class DQN(BaseDQN):
    def __init__(self):
        super(DQN, self).__init__()
        self.eval_net = BaseModel(NUM_STATES, NUM_ACTIONS).to(device)
        self.target_net = BaseModel(NUM_STATES, NUM_ACTIONS).to(device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def evaluating(self, samples: List[Data]) -> Tuple[torch.Tensor, torch.Tensor]:
        states, actions, rewards, next_states, non_ends = samples_to_tensors(samples)

        with torch.no_grad():
            target_next_q: torch.Tensor = self.target_net(next_states)
            next_qs = target_next_q.max(1)[0]
            targets = rewards + GAMMA * next_qs * non_ends

        qs = self.eval_net(states)
        preds = qs[torch.arange(BATCH_SIZE), actions]
        return preds, targets


class DoubleDQN(BaseDQN):
    def __init__(self):
        super(DoubleDQN, self).__init__()
        self.eval_net = BaseModel(NUM_STATES, NUM_ACTIONS).to(device)
        self.target_net = BaseModel(NUM_STATES, NUM_ACTIONS).to(device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def evaluating(self, samples: List[Data]) -> Tuple[Tensor, Tensor]:
        states, actions, rewards, next_states, non_ends = samples_to_tensors(samples)

        with torch.no_grad():
            eval_next_q: torch.Tensor = self.eval_net(next_states)
            next_actions = eval_next_q.max(1)[1]
            target_next_q: torch.Tensor = self.target_net(next_states)
            next_qs = target_next_q[torch.arange(BATCH_SIZE), next_actions]
            targets = rewards + GAMMA * next_qs * non_ends

        eval_q = self.eval_net(states)
        preds = eval_q[torch.arange(BATCH_SIZE), actions]
        return preds, targets


class DuelingDQN(BaseDQN):
    def __init__(self):
        super(DuelingDQN, self).__init__()
        self.eval_net = DuelingModel(NUM_STATES, NUM_ACTIONS).to(device)
        self.target_net = DuelingModel(NUM_STATES, NUM_ACTIONS).to(device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, EPSILON=1.0):
        state = torch.tensor(state, dtype=torch.float).to(device)
        if np.random.random() > EPSILON:  # random number
            # greedy policy
            with torch.no_grad():
                _, action_value = self.eval_net(state)
            action = torch.argmax(action_value).item()
        else:
            # random policy
            action = np.random.randint(0, NUM_ACTIONS)  # int random number
        return action

    def evaluating(self, samples: List[Data]) -> Tuple[Tensor, Tensor]:
        states, actions, rewards, next_states, non_ends = samples_to_tensors(samples)

        with torch.no_grad():
            target_next_v, target_next_q = self.target_net(next_states)
            next_qs = target_next_v + target_next_q - target_next_q.mean(dim=1, keepdim=True)
            next_qs = next_qs.max(1)[0]
            targets = rewards + GAMMA * next_qs * non_ends

        eval_v, eval_q = self.eval_net(states)
        eval_q = eval_v + eval_q - eval_q.mean(dim=1, keepdim=True)
        preds = eval_q[torch.arange(BATCH_SIZE), actions]
        return preds, targets


class DuelingDoubleDQN(BaseDQN):
    def __init__(self):
        super(DuelingDoubleDQN, self).__init__()
        self.eval_net = DuelingModel(NUM_STATES, NUM_ACTIONS).to(device)
        self.target_net = DuelingModel(NUM_STATES, NUM_ACTIONS).to(device)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, state, EPSILON=1.0):
        state = torch.tensor(state, dtype=torch.float).to(device)
        if np.random.random() > EPSILON:  # random number
            # greedy policy
            with torch.no_grad():
                _, action_value = self.eval_net.forward(state)
            action = torch.argmax(action_value).item()
        else:
            # random policy
            action = np.random.randint(0, NUM_ACTIONS)  # int random number
        return action

    def evaluating(self, samples: List[Data]) -> Tuple[Tensor, Tensor]:
        states, actions, rewards, next_states, non_ends = samples_to_tensors(samples)

        with torch.no_grad():
            eval_next_v, eval_next_q = self.eval_net(next_states)
            eval_next_q = eval_next_v + eval_next_q - eval_next_q.mean(dim=1, keepdim=True)
            next_actions = eval_next_q.max(1)[1]

            target_next_v, target_next_q = self.target_net(next_states)
            next_qs = target_next_v + target_next_q - target_next_q.mean(dim=1, keepdim=True)
            next_qs = next_qs[torch.arange(BATCH_SIZE), next_actions]
            targets = rewards + GAMMA * next_qs * non_ends

        eval_v, eval_q = self.eval_net(states)
        eval_q = eval_v + eval_q - eval_q.mean(dim=-1, keepdim=True)
        preds = eval_q[torch.arange(BATCH_SIZE), actions]
        return preds, targets


def main(model):
    if TEST:
        log_dir = SAVE_PATH_PREFIX  # os.path.dirname(SAVE_PATH_PREFIX)
        last_dir = sorted(os.listdir(log_dir))[-1]
        pth_dir = os.path.join(log_dir, last_dir, "ckpt")
        pth_file = glob.glob(os.path.join(pth_dir, "best-*.pth"))[-1]
        model.load_net(pth_file)
    else:
        memory = Memory(MEMORY_CAPACITY)
        writer = SummaryWriter(f"{SAVE_PATH_PREFIX}")

    best_reward = 0
    for i in range(EPISODES if not TEST else 1):
        print("EPISODE: ", i)
        state, info = env.reset(seed=SEED)

        ep_reward = 0
        num_step = 0
        done = False
        # 后一个条件是避免同一个episode的把memory的数据占满，导致不是iid的数据，从而训练效果糟糕
        while not done and (num_step <= 0.1 * MEMORY_CAPACITY or TEST):
            num_step += 1
            # choose best action
            action = model.choose_action(state=state, EPSILON=EPSILON if not TEST else 0)
            # observe next state and reward
            next_state, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            if TEST:
                env.render()
            else:
                memory.set(Data(state, action, reward, next_state, i, done))
                if memory.size() >= MIN_CAPACITY:
                    samples = memory.get(BATCH_SIZE)
                    model.learn(samples)
            state = next_state

        if not TEST:
            if ep_reward < 30:
                memory.remove_last()
            elif ep_reward > best_reward:
                best_reward = ep_reward
                model.save_train_model(f"best-{i}")
            writer.add_scalar("reward", ep_reward, global_step=i)

        print(f"episode: {i} , the episode reward is {round(ep_reward, 3)}")


if __name__ == "__main__":
    # hyper-parameters
    EPISODES = 500  # 训练/测试幕数
    BATCH_SIZE = 800
    LR = 0.00025
    GAMMA = 0.98
    SAVING_IETRATION = 1000  # 保存Checkpoint的间隔
    MEMORY_CAPACITY = 10000  # Memory的容量
    MIN_CAPACITY = 1000  # 开始学习的下限
    Q_NETWORK_ITERATION = 10  # 同步target network的间隔
    EPSILON = 0.01  # epsilon-greedy
    SEED = 0
    MODEL_PATH = ""
    SAVE_PATH_PREFIX = os.path.join("log", "CartPole")
    TEST = True

    env = gym.make("CartPole-v1", render_mode="rgb_array" if TEST else None)
    env = RecordVideo(env, video_folder=f"{SAVE_PATH_PREFIX}/video", disable_logger=True)
    # env = gym.make('MountainCar-v0', render_mode="human" if TEST else None)
    # env = gym.make("LunarLander-v2",continuous=False,gravity=-10.0,enable_wind=True,wind_power=15.0,turbulence_power=1.5,render_mode="human" if TEST else None)

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUM_ACTIONS = env.action_space.n  # 2
    NUM_STATES = env.observation_space.shape[0]  # 4
    ENV_A_SHAPE = (
        0  # 0, to confirm the shape
        if np.issubdtype(type(env.action_space.sample()), int)
        else env.action_space.sample().shape
    )

    model = DuelingDoubleDQN()
    SAVE_PATH_PREFIX = os.path.join(SAVE_PATH_PREFIX, model.name())
    if not TEST:
        SAVE_PATH_PREFIX = os.path.join(SAVE_PATH_PREFIX, str(time.time()))
        os.makedirs(f"{SAVE_PATH_PREFIX}/ckpt", exist_ok=True)

    main(model)
