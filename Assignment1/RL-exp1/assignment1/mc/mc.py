import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

#create env
env = BlackjackEnv()

def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.
    
    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.
    
    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.
    
    """
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def mc(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.
    
    Args:
        env: OpenAI gym environment.
        num_episodes: Number of episodes to sample.
        discount_factor: Gamma discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.
    
    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function that takes an observation as an argument and returns
        action probabilities
    """
    
    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(1, num_episodes + 1):
        # Print out which episode we're on, useful for debugging.
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

#############################################Implement your code###################################################################################################
        # step 1 : Generate an episode.
            # An episode is an array of (state, action, reward) tuples
        def generate_one_episode(env, generate_policy):
            trajectory = []
            state = env.reset()
            for i in range(1000):
                Pi_table = generate_policy(state)
                action = np.random.choice(np.arange(len(Pi_table)), p=Pi_table)
                next_state, reward, done, _ = env.step(action)
                trajectory.append((next_state, action, reward))
                if done:
                    break
                state = next_state
            return trajectory
        trajectory = generate_one_episode(env, policy)
        
        # step 2 : Find all (state, action) pairs we've visited in this episode
        s_a_pairs = set([(x[0], x[1]) for x in trajectory])
        
        # step 3 : Calculate average return for this state over all sampled episodes
        
        # first visited version
        # for state, action in s_a_pairs:
        #     s_a = (state, action)
        #     first_visit_id = next(i for i, x in enumerate(trajectory) if x[0] == state and x[1] == action)
        #     G = sum([x[2] * (discount_factor ** i) for i, x in enumerate(trajectory[first_visit_id:])])
        #     returns_sum[s_a] += G
        #     returns_count[s_a] += 1.
        #     Q[state][action] = returns_sum[s_a] / returns_count[s_a]
        
        # # every visited version
        for state, action in s_a_pairs:
            s_a = (state, action)
            G = 0
            all_visit_id = [i for i, x in enumerate(trajectory) if x[0] == state and x[1] == action]
            for visit_id in all_visit_id:
                G += sum([x[2] * np.power(discount_factor, i) for i, x in enumerate(trajectory[visit_id:])])
            G /= len(all_visit_id)
            returns_sum[s_a] += G
            returns_count[s_a] += 1.
            Q[state][action] = returns_sum[s_a] / returns_count[s_a]
        
 #############################################Implement your code end###################################################################################################
   
    return Q, policy

Q, policy = mc(env, num_episodes=500000, epsilon=0.1)

# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")