#!/usr/bin/env python
# coding: utf-8

# # Please install the following python libraries
# - python3: https://www.python.org/
# - numpy: https://numpy.org/install/
# - tqdm: https://github.com/tqdm/tqdm#installation
# - matplotlib: https://matplotlib.org/stable/users/installing/index.html
# 
# If you encounter the error: "IProgress not found. Please update jupyter & ipywidgets"
#     
# Please install the ipywidgets as follows:
# 
#     with pip, do
#     - pip install ipywidgets
#     
#     with conda, do
#     - conda install -c conda-forge ipywidgets
#     
# Restart your notebook 

# In[ ]:


import random
import numpy as np
import matplotlib.pyplot as plt
import tqdm


# # Q3: Solving Four Rooms using semi-gradient SARSA with state aggregation
# 

# In[ ]:


""" Four Rooms Environment Implementation
"""
class FourRooms(object):
    def __init__(self):
        # We define the grid for the Four Rooms domain
        self.grid = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])

        # We define the observation space consisting of all empty cells
        # Note: We have to flip the coordinates from (row_idx, column_idx) -> (x, y),
        # where x = column_idx, y = 10 - row_idx
        self.observation_space = np.argwhere(self.grid == 0.0).tolist()  # Fine all empty cells
        self.observation_space = self.arr_coords_to_four_room_coords(self.observation_space)

        # We define the action space
        self.action_space = {'up': np.array([0, 1]),
                             'down': np.array([0, -1]),
                             'left': np.array([-1, 0]),
                             'right': np.array([1, 0])}
        self.action_names = ['up', 'down', 'left', 'right']

        # We define the start location
        self.start_location = [0, 0]

        # We define the goal location
        self.goal_location = [10, 10]

        # We find all wall cells
        self.walls = np.argwhere(self.grid == 1.0).tolist()  # find all wall cells
        self.walls = self.arr_coords_to_four_room_coords(self.walls)  # convert to Four Rooms coordinates

        # This is an episodic task, we define a timeout: maximal time steps = 459
        self.max_time_steps = 459

        # We define other useful variables
        self.agent_location = None  # track the agent's location in one episode.
        self.action = None  # track the agent's action
        self.t = 0  # track the current time step in one episode

    @staticmethod
    def arr_coords_to_four_room_coords(arr_coords_list):
        """
        Function converts the array coordinates to the Four Rooms coordinates (i.e, The origin locates at bottom left).
        E.g., The coordinates (0, 0) in the numpy array is mapped to (0, 10) in the Four Rooms coordinates.
        Args:
            arr_coords_list (list): a list variable consists of tuples of locations in the numpy array

        Return:
            four_room_coords_list (list): a list variable consists of tuples of converted locations in the
                                          Four Rooms environment.
        """
        # Note: We have to flip the coordinates from (row_idx, column_idx) -> (x, y),
        # where x = column_idx, y = 10 - row_idx
        four_room_coords_list = [(column_idx, 10 - row_idx) for (row_idx, column_idx) in arr_coords_list]
        return four_room_coords_list

    def reset(self):
        # We reset the agent's location to the start location
        self.agent_location = self.start_location

        # We reset the timeout tracker to be 0
        self.t = 0

        # We set the information
        info = {}
        return self.agent_location, info

    def step(self, action):
        """
        Args:
            action (string): a string variable (i.e., "UP"). All feasible values are ["up", "down", "left", "right"].
        """
        # With probability 0.8, the agent takes the correct direction.
        # With probability 0.2, the agent takes one of the two perpendicular actions.
        # For example, if the correct action is "LEFT", then
        #     - With probability 0.8, the agent takes action "LEFT";
        #     - With probability 0.1, the agent takes action "UP";
        #     - With probability 0.1, the agent takes action "DOWN".
        if np.random.uniform() < 0.2:
            if action == "left" or action == "right":
                action = np.random.choice(["up", "down"], 1)[0]
            else:
                action = np.random.choice(["right", "left"], 1)[0]

        # Convert the agent's location to array
        loc_arr = np.array(self.agent_location)

        # Convert the action name to movement array
        act_arr = self.action_space[action]

        # Compute the agent's next location
        next_agent_location = np.clip(loc_arr + act_arr,
                                      a_min=np.array([0, 0]),
                                      a_max=np.array([10, 10])).tolist()

        # Check if the agent crashes into walls, it stays at the current location.
        if tuple(next_agent_location) in self.walls:
            next_agent_location = self.agent_location

        # Compute the reward
        reward = 1.0 if next_agent_location == self.goal_location else 0.0

        # Check the termination
        # If the agent reaches the goal, reward = 1, done = True
        # If the time steps reaches the maximal number, reward = 0, done = True.
        if reward == 1.0 or self.t == self.max_time_steps:
            terminated = True
        else:
            terminated = False

        # Update the agent's location, action and time step trackers
        self.agent_location = next_agent_location
        self.action = action
        self.t += 1

        return next_agent_location, reward, terminated, False, {}

    def render(self):
        # plot the agent and the goal
        # empty cell = 0
        # wall cell = 1
        # agent cell = 2
        # goal cell = 3
        plot_arr = self.grid.copy()
        plot_arr[10 - self.agent_location[1], self.agent_location[0]] = 2
        plot_arr[10 - self.goal_location[1], self.goal_location[0]] = 3
        plt.clf()
        plt.title(f"state={self.agent_location}, act={self.action}")
        plt.imshow(plot_arr)
        plt.show(block=False)
        plt.pause(0.1)

    @staticmethod
    def test():
        my_env = FourRooms()
        state, _ = my_env.reset()

        for _ in range(100):
            action = np.random.choice(list(my_env.action_space.keys()), 1)[0]

            next_state, reward, done, _, _ = my_env.step(action)
            my_env.render()

            if done:
                state, _ = my_env.reset()
            else:
                state = next_state


# In[ ]:


def plot_curves(arr_list, legend_list, color_list, ylabel, fig_title):
    """
    Args:
        arr_list (list): list of results arrays to plot
        legend_list (list): list of legends corresponding to each result array
        color_list (list): list of color corresponding to each result array
        ylabel (string): label of the Y axis

        Note that, make sure the elements in the arr_list, legend_list and color_list are associated with each other correctly.
        Do not forget to change the ylabel for different plots.
    """
    # set the figure type
    fig, ax = plt.subplots(figsize=(12, 8))

    # PLEASE NOTE: Change the labels for different plots
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Time Steps")

    # ploth results
    h_list = []
    for arr, legend, color in zip(arr_list, legend_list, color_list):
        # compute the standard error
        arr_err = arr.std(axis=0) / np.sqrt(arr.shape[0])
        # plot the mean
        h, = ax.plot(range(arr.shape[1]), arr.mean(axis=0), color=color, label=legend)
        # plot the confidence band
        arr_err *= 1.96
        ax.fill_between(range(arr.shape[1]), arr.mean(axis=0) - arr_err, arr.mean(axis=0) + arr_err, alpha=0.3,
                        color=color)
        # save the plot handle
        h_list.append(h)

    # plot legends
    ax.set_title(f"{fig_title}")
    ax.legend(handles=h_list)

    # save the figure
    plt.savefig(f"{fig_title}.png", dpi=200)

    plt.show()


# # Q3 - (a): Implement the semi-gradient SARSA
# 
# As described in the question, you are asked to implement the semi-gradient SARSA with a very simple state aggregation strategy. That is aggregating both states and actions to itself. Indeed, this will have similar results as applying SARSA directly. 
# 
# **Please implement the following state aggregation strategy**
# 
# - For each state, its aggregated state is itself. E.g. [0, 0] is aggregated to [0, 0] only. 
# - For each action, its aggregated action is also itself.

# In[ ]:


class SemiGradientSARSAAgent(object):
    def __init__(self, env, info):
        """
        Function to initialize the semi-gradient SARSA agent
        Args:
            env: the environment that the agent interacts with
            info (dict): a dictionary variable contains all necessary parameters.

        Note that: In this question, we fix the function approximation and only discuss different
                   state aggregation strategies. Specifically, we design the following function approximation:
                   1. Feature: we use the one-hot encoding to compute the feature for each state-action pair.
                               E.g., state = [0, 0] and action = "Up" will correspond to a unique one-hot representation
                                     [0, 0, 0, 1, 0, ..., 0].
                   2. Function approximation: we use the linear function approximation. Specifically, the approximation
                      function is represented by an N x 1 weight vector, where N = |S| * |A|.

        Importantly, as described in the question, we only aggregate the states.
        """
        # Store the environment
        self.env = env

        """ Learning parameters for semi-gradient SARSA """
        # Store the number of learning episodes
        self.episode_num = info['episode_num']

        # Store the Q-learning step size alpha
        self.alpha = info['alpha']

        # Store the discount factor
        self.gamma = info['gamma']

        # Initialize the epsilon
        self.epsilon = info['epsilon']

        # Store the hyerparameters
        self.params = info

        """ State aggregation for semi-gradient SARSA """
        # Compute the total number of state after the aggregation
        self.state_space, self.state_num = self.create_state_aggregation()

        # Compute the total number of the actions
        self.action_num = len(self.env.action_names)

        """ Function approximation for semi-gradient SARSA"""
        # We consider a linear function approximation here and store all the weights here
        self.weights_fn = np.zeros((self.state_num * self.action_num))

        # We construct a numpy array that contains the one-hot features for all states.
        # The size is (|S| * |A|) x (|S| * |A|)
        self.feature_arr = np.eye(self.state_num * self.action_num)

    def create_state_aggregation(self):
        """
        Function that returns the aggregated state space and the number of the aggregated states.
        """
        """CODE HERE: your state aggregation strategy. Hint: you can start with a simple state aggregation
           that just aggregate each state to itself. In other words, the aggregated state space is just
           the original state space.
           
           You have to return:
            1. the aggregated state space (Any data structure that you find it easier to render the index of the 
               aggregated state.)
            2. the number of the aggregated states (int)
        """
        # Create a dictionary to map states to indices:
        state_to_index = {state: i for i, state in enumerate(self.env.observation_space)}
        return state_to_index, len(self.env.observation_space)  # Return the mapping and size

    def _aggregate_state_idx(self, state):
        """
        Function returns the index of aggregated state given an original state

        Args:
            state (list): original state
        """
        """CODE HERE: based on your state aggregation, return the index of the aggregated state given an original
           state. 
           
           You have to return:
           1. index (int) of the aggregated state given the original state
        """
        try:
            return self.state_space[tuple(state)]  # Use the dictionary for lookup
        except KeyError:  # Handle invalid states
            return -1

    def _aggregate_action_idx(self, action):
        """
        Function returns the index of aggregated action.
        Args:
            action (string): name of the action

        To be simple, here, one action only aggregates to itself
        """
        return self.env.action_names.index(action)

    def _get_state_action_feature(self, state, action):
        """
        Function that returns the one-hot feature given a state-action pair.

        Args:
            state (list): original state
            action (string): name of the action
        """
        # Get the unique index of the aggregated state
        state_index = self._aggregate_state_idx(state)
        # Get the unique index of the aggregated action
        action_index = self._aggregate_action_idx(action)
        # Compute the state(aggregated)-action index
        state_action_index = self.state_num * action_index + state_index
        # Get the one-hot feature of the state
        return self.feature_arr[state_action_index]

    def function_approximation(self, state, action):
        """
        Function that computes the Q value given a state-action pair using linear function approximation.
        Args:
            state (list): original state
            action (string): name of the action
        """
        state_action_feature = self._get_state_action_feature(state, action)
        return np.matmul(state_action_feature.T, self.weights_fn)

    def render_q_value(self, state, action):
        """
        Function that returns the Q value given a state-action pair

        Args:
            state (list): original state
            action (string): name of the action
        """
        return self.function_approximation(state, action)

    def epsilon_greedy_policy(self, state):
        """
        Function implements the epsilon-greedy policy
        Args:
            state (list): original state
        """
        """CODE HERE: implement the epsilon-greedy policy using function approximation. Break ties if happens """
        if np.random.random() < self.epsilon:
            return np.random.choice(self.env.action_names)
        else:
            q_values = [self.function_approximation(state, a) for a in self.env.action_names]
            max_q = np.max(q_values)  # Find the maximum Q-value
            best_actions = [a for a, q in zip(self.env.action_names, q_values) if q == max_q]  # Find all actions with max_q
            return np.random.choice(best_actions)  # Randomly choose among the best actions
        
    def update_weights(self, s, a, r, s_prime, a_prime):
        """
        Function that updates the weights using semi-gradients

        Args:
            s (list): original state
            a (string): action name
            r (float): reward
            s_prime (list): original next state
            a_prime (string): next action name
        """
        """ CODE HERE: implement the update of the semi-gradient SARSA """
        q_s_a = self.function_approximation(s, a)
        state_action_feature = self._get_state_action_feature(s, a)

        if s_prime is None:  # Terminal state
            target = r
        else:
            target = r + self.gamma * self.function_approximation(s_prime, a_prime)
            
        self.weights_fn += self.alpha * (target - q_s_a) * state_action_feature

    def run(self):
        # Save the discounted return for each episode
        discounted_returns = []

        # Semi-gradient SARSA starts
        for ep in tqdm.trange(self.episode_num):
            """CODE HERE: Implement the pseudocode of the Semi-gradient SARSA"""
            # Reset the agent to initial STATE at the beginning of every episode
            state, _ = self.env.reset()
            state = list(state)

            # Render an ACTION based on the initial STATE
            action = self.epsilon_greedy_policy(state)

            # Store rewards to compute return G for the current episode.
            reward_list = []
            
            # Loop the episode
            for t in range(self.env.max_time_steps):
                # Take the ACTION and observe REWARD and NEXT STATE
                next_state, reward, done, _, _ = self.env.step(action)
                next_state = list(next_state)
                
                if done:  # Use 'done' flag to detect terminal state
                    next_state = None 
                    next_action = None
                else:
                    next_action = self.epsilon_greedy_policy(next_state)

                self.update_weights(state, action, reward, next_state, next_action)
                
                """DO NOT CHANGE BELOW"""
                # Save the reward for plotting
                reward_list.append(reward)

                # Reset the environment
                if done:
                    break
                else:
                    state = next_state
                    action = next_action

            # compute the discounted return for the current episode
            G = 0
            for reward in reversed(reward_list):
                G = reward + self.gamma * G
            discounted_returns.append(G)

        return discounted_returns
           


# In[ ]:


if __name__ == "__main__":
    # set the random seed
    np.random.seed(1234)
    random.seed(1234)

    # set hyper-parameters
    params = {
        "episode_num": 100,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.1,
        'tile_size': 1
    }

    # set running trials
    run_trial = 10

    results_1 = []
    for _ in range(run_trial):
        # create the environment
        my_env = FourRooms()

        # run semi-gradient SARSA with tile n = 2
        tabular_sarsa = SemiGradientSARSAAgent(my_env, params)
        res = tabular_sarsa.run()

        # save result for each running trial
        results_1.append(np.array(res))

    plot_curves([np.array(results_1)],
                ["semi-gradient SARSA"],
                ["b"],
                "Averaged discounted return", "Q3 - (a): semi-gradient SARSA")


# # Q3 - (b) [5180]: Implement the semi-gradient SARSA with Tile-based/Room-based aggregation.
# 
# As described in the question, you are asked to implement the semi-gradient SARSA with **Tile-based/Room-based** state aggregation strategy. That is grouping the nearby states in a n x n (i.e., n = 2) tile as one aggregated state. As for the function approximation, we assume to use the same as above.  
# 
# **Plot**: Plot the learning curves of tile size n = 2 and Room-based aggregation in the same plot. You can use the plot function above to generate the plot. 

# In[ ]:


""" Implement the Tile-based Agent here. We inherit it from the "SemiGradientSARSAAgent" above
"""
class TileAgent(SemiGradientSARSAAgent):
    def __init__(self, env, info):
        """
        Function to initialize the semi-gradient SARSA agent
        Args:
            env: the environment that the agent interacts with
            info (dict): a dictionary variable contains all necessary parameters.

        Note that: In this question, we will implement a simple state aggregation strategies.
                   Specifically, we design the following function approximation:
                   1. Feature: we use the one-hot encoding to compute the feature for each state-action pair.
                               E.g., state = [0, 0] and action = "Up" (state-action pair) will correspond to
                               a unique one-hot representation $f(s, a) = [0, 0, 0, 1, 0, ..., 0]$.
                   2. Weights: we define a weight vector $w$ having the sample shape as the feature vector.
                               Specifically, the Q(s, a) can be estimated by Q(s, a) = w^{T} * f(s, a)

        Importantly, as described in the question, we only aggregate the states.
        """
        super().__init__(env, info)
        
        """ State aggregation for semi-gradient SARSA """
        # Compute the total number of state after the aggregation
        self.state_space, self.state_num = self.create_state_aggregation()

        # Compute the total number of the actions
        self.action_num = len(self.env.action_names)

        """ Function approximation for semi-gradient SARSA"""
        # We create a weight with shape |S| * |A|
        self.weights_fn = np.zeros((self.state_num * self.action_num))

        # We construct a numpy array that contains the one-hot features for all state-action pairs.
        # The size is (|S| * |A|) x (|S| * |A|).
        # Each i-th row is the one-hot encoding for state-action pair with index i.
        self.feature_arr = np.eye(self.state_num * self.action_num)

    def create_state_aggregation(self):
        """
        Function that returns the aggregated state space and the number of the aggregated states.
        """
        
        """CODE HERE: Implement the Tile-based state aggregation here.
           Hint: you can manually discretize the original states using the Tile-based method.
           For example, you can copy the grid from the Four Rooms environment
           and manually aggregate the states (value = 0) in the grid. 
           
           You have to return:
            1. the aggregated state space (Any data structure that you find it easier to render the index of the 
               aggregated state.)
            2. the number of the aggregated states (int)
        """
        
        tile_size = self.params['tile_size']
        grid_size = 11
        aggregated_state_space = {}
        state_index = 0

        for i in range(0, grid_size, tile_size):
            for j in range(0, grid_size, tile_size):
                aggregated_state = []
                for x in range(i, min(i + tile_size, grid_size)):
                    for y in range(j, min(j + tile_size, grid_size)):
                        if (x, y) in self.env.observation_space:
                            aggregated_state.append((x, y))
                if aggregated_state:
                    aggregated_state_space[tuple(aggregated_state)] = state_index
                    state_index += 1

        return aggregated_state_space, state_index 

    def _aggregate_state_idx(self, state):
        """
        Function returns the index of aggregated state given an original state

        Args:
            state (list): original state
        """
        """CODE HERE: based on your state aggregation, return the index of the aggregated state given an original
           state. 
           
           You have to return:
           1. index (int) of the aggregated state given the original state
        """
        
        # render the index of the aggregated state
        for aggregated_state, idx in self.state_space.items():
            if tuple(state) in aggregated_state:
                return idx
        return -1


# In[ ]:


""" Implement the Room-based Agent here. We inherit it from the "SemiGradientSARSAAgent" above
"""
class RoomAgent(SemiGradientSARSAAgent):
    def __init__(self, env, info):
        """
        Function to initialize the semi-gradient SARSA agent
        Args:
            env: the environment that the agent interacts with
            info (dict): a dictionary variable contains all necessary parameters.

        Note that: In this question, we will implement a simple state aggregation strategies.
                   Specifically, we design the following function approximation:
                   1. Feature: we use the one-hot encoding to compute the feature for each state-action pair.
                               E.g., state = [0, 0] and action = "Up" (state-action pair) will correspond to
                               a unique one-hot representation $f(s, a) = [0, 0, 0, 1, 0, ..., 0]$.
                   2. Weights: we define a weight vector $w$ having the sample shape as the feature vector.
                               Specifically, the Q(s, a) can be estimated by Q(s, a) = w^{T} * f(s, a)

        Importantly, as described in the question, we only aggregate the states.
        """
        super().__init__(env, info)
        
        """ State aggregation for semi-gradient SARSA """
        # Compute the total number of state after the aggregation
        self.state_space, self.state_num = self.create_state_aggregation()

        # Compute the total number of the actions
        self.action_num = len(self.env.action_names)

        """ Function approximation for semi-gradient SARSA"""
        # We create a weight with shape |S| * |A|
        self.weights_fn = np.zeros((self.state_num * self.action_num))

        # We construct a numpy array that contains the one-hot features for all state-action pairs.
        # The size is (|S| * |A|) x (|S| * |A|).
        # Each i-th row is the one-hot encoding for state-action pair with index i.
        self.feature_arr = np.eye(self.state_num * self.action_num)

    def create_state_aggregation(self):
        """
        Function that returns the aggregated state space and the number of the aggregated states.
        """
        """CODE HERE: your state aggregation strategy. Hint: you can start with a simple state aggregation
           that just aggregate each state to itself. In other words, the aggregated state space is just
           the original state space.
           
           You have to return:
            1. the aggregated state space (Any data structure that you find it easier to render the index of the 
               aggregated state.)
            2. the number of the aggregated states (int)
        """

        room1 = []
        room2 = []
        room3 = []
        room4 = []

        for state in self.env.observation_space:
            if state[0] <= 4 and state[1] <= 4:
                room1.append(state)
            elif state[0] >= 6 and state[1] <= 4:
                room2.append(state)
            elif state[0] <= 4 and state[1] >= 6:
                room3.append(state)
            elif state[0] >= 6 and state[1] >= 6:
                room4.append(state)

        aggregated_state_space = [tuple(room1), tuple(room2), tuple(room3), tuple(room4)]
        aggregate_state_num = len(aggregated_state_space)

        return aggregated_state_space, aggregate_state_num

    def _aggregate_state_idx(self, state):
        """
        Function returns the index of aggregated state given an original state

        Args:
            state (list): original state
        """
        """CODE HERE: based on your state aggregation, return the index of the aggregated state given an original
           state. 
           
           You have to return:
           1. index (int) of the aggregated state given the original state
        """
        for idx, room in enumerate(self.state_space):
            if tuple(state) in room:
                return idx
        return -1


# In[ ]:


if __name__ == "__main__":
    # set the random seed
    np.random.seed(1234)
    random.seed(1234)

    # set hyper-parameters
    params = {
        "episode_num": 100,
        "alpha": 0.1,
        "gamma": 0.99,
        "epsilon": 0.1,
        "tile_size": 2
    }

    # set running trials; You can trial run_trial = 5 to debug
    run_trial = 10

    # run experiment for the Tile-based method
    results_tile = []
    for _ in range(run_trial):        
        # create the environment
        my_env = FourRooms()

        # run semi-gradient SARSA with Tile-based method with tile size n = 2
        my_agent = TileAgent(my_env, params)
        res = my_agent.run()

        # save result for each running trial
        results_tile.append(np.array(res))
        
    # run experiment for the Room-based method
    results_room = []
    for _ in range(run_trial):        
        # create the environment
        my_env = FourRooms()

        # run semi-gradient SARSA with Room-based method
        my_agent = RoomAgent(my_env, params)
        res = my_agent.run()

        # save result for each running trial
        results_room.append(np.array(res))


# In[ ]:


# plot the results
plot_curves([np.array(results_1), np.array(results_tile), np.array(results_room)],
            ["State-aggregation: identical", "State-aggregation: Tile = 2x2", "State aggregation: Room-based"],
            ["b", "r", "g"],
            "Averaged discounted return", "Q3 - (b): Comparison between three state aggregation strategies")


# # Q3 - (d):  Adapt your implementation of semi-gradient one-step SARSA for linear function approximation

# In[ ]:


"""Implement your code here."""
class LinearApproxSARSAAgent(SemiGradientSARSAAgent):
    def __init__(self, env, info):
        super().__init__(env, info)

        self.weights_fn = np.zeros(12)

    def _get_state_action_feature(self, state, action):
        if state is None:
            return np.zeros(12)
        x, y = state
        action_idx = self.env.action_names.index(action)
        feature = np.zeros(12)

        feature[action_idx * 3] = x
        feature[action_idx * 3 + 1] = y
        feature[action_idx * 3 + 2] = 1  # Bias term

        return feature


# In[ ]:


if __name__ == "__main__":
    # set the random seed
    np.random.seed(1234)
    random.seed(1234)

    # set hyper-parameters
    params = {
        "episode_num": 100,
        "alpha": 0.01, # Reduced learning rate
        "gamma": 0.99,
        "epsilon": 0.1,
        "tile_size": 2 
    }

    # set running trials; You can trial run_trial = 5 to debug
    run_trial = 10

    # run experiment for linear function approximation
    results_linear = []
    for _ in range(run_trial):
        my_env = FourRooms()
        my_agent = LinearApproxSARSAAgent(my_env, params)
        res = my_agent.run()
        results_linear.append(np.array(res))

    plot_curves([np.array(results_1), np.array(results_tile), np.array(results_room), np.array(results_linear)],
                ["State-aggregation: identical", "State-aggregation: Tile = 2x2", "State-aggregation: Room-based", "Linear Function Approximation"],
                ["b", "r", "g", "purple"], 
                "Averaged discounted return", "Q3 - (d): Comparison between four state aggregation strategies")


# # Q3 - (e) [5180]: Implement the following two features, and plot the learning curves.

# In[ ]:


"""Implement your code here."""
class NormalizedLinearApproxSARSAAgent(SemiGradientSARSAAgent):
    def __init__(self, env, info):
        super().__init__(env, info)

        self.weights_fn = np.zeros(12)

    def _get_state_action_feature(self, state, action):
        if state is None:
            return np.zeros(12)
        x, y = state
        action_idx = self.env.action_names.index(action)
        feature = np.zeros(12)

        feature[action_idx * 3] = (x - 10) / 10
        feature[action_idx * 3 + 1] = (y - 10) / 10
        feature[action_idx * 3 + 2] = 1  # Bias term

        return feature


# In[ ]:


if __name__ == "__main__":
    # set the random seed
    np.random.seed(1234)
    random.seed(1234)

    # set hyper-parameters
    params = {
        "episode_num": 100,
        "alpha": 0.01, # Reduced learning rate
        "gamma": 0.99,
        "epsilon": 0.1,
        "tile_size": 2
    }

    # set running trials; You can trial run_trial = 5 to debug
    run_trial = 10

    # run experiment for linear function approximation
    results_normalized_linear = []
    for _ in range(run_trial):
        my_env = FourRooms()
        my_agent = NormalizedLinearApproxSARSAAgent(my_env, params)
        res = my_agent.run()
        results_normalized_linear.append(np.array(res))


# In[ ]:


"""The distance matrix is here"""

distance_matrix = np.array([[14, 13, 12, 11, 10, -1,  4,  3,  2,  1,  0],
                            [13, 12, 11, 10,  9, -1,  5,  4,  3,  2,  1],
                            [12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2],
                            [13, 12, 11, 10,  9, -1,  7,  6,  5,  4,  3],
                            [14, 13, 12, 11, 10, -1,  8,  7,  6,  5,  4],
                            [-1, 14, -1, -1, -1, -1,  9,  8,  7,  6,  5],
                            [16, 15, 16, 17, 18, -1, -1, -1,  8, -1, -1],
                            [17, 16, 17, 18, 17, -1, 11, 10,  9, 10, 11],
                            [18, 17, 18, 17, 16, -1, 12, 11, 10, 11, 12],
                            [19, 18, 17, 16, 15, 14, 13, 12, 11, 12, 13],
                            [20, 19, 18, 17, 16, -1, 14, 13, 12, 13, 14]])
plt.imshow(distance_matrix)
plt.show()


# In[ ]:


"""Implement your code here."""
class DistanceLinearApproxSARSAAgent(SemiGradientSARSAAgent):
    def __init__(self, env, info):
        super().__init__(env, info)

        self.weights_fn = np.zeros(16)

    def _get_state_action_feature(self, state, action):
        if state is None:
            return np.zeros(16)
        x, y = state
        action_idx = self.env.action_names.index(action)
        feature = np.zeros(16)
        
        max_distance = 20  
        scaled_distance = min(distance_matrix[x, y], max_distance) / max_distance

        feature[action_idx * 4] = scaled_distance
        feature[action_idx * 4 + 1] = (x - 10) / 10 
        feature[action_idx * 4 + 2] = (y - 10) / 10 
        feature[action_idx * 4 + 3] = 1.0  # Bias
        return feature


# In[ ]:


if __name__ == "__main__":
    # set the random seed
    np.random.seed(1234)
    random.seed(1234)

    # set hyper-parameters
    params = {
        "episode_num": 100,
        "alpha": 0.01, # Reduced learning rate
        "gamma": 0.99,
        "epsilon": 0.1,
        "tile_size": 2
    }

    # set running trials; You can trial run_trial = 5 to debug
    run_trial = 10

    # run experiment for linear function approximation
    results_distance_linear = []
    for _ in range(run_trial):
        my_env = FourRooms()
        my_agent = DistanceLinearApproxSARSAAgent(my_env, params)
        res = my_agent.run()
        results_distance_linear.append(np.array(res))


# In[ ]:


plot_curves([np.array(results_normalized_linear), np.array(results_distance_linear)],
            ["Normalized Linear Function Approximation", "Distance Linear Function Approximation"],
            ["b", "r"],
            "Averaged discounted return", "Q3 - (e): Comparison between two state aggregation strategies")


# # Q4: Mountain car.
# ### In this problem, you need to use the mountain car env from gymnasium
# link: https://gymnasium.farama.org/environments/classic_control/mountain_car/
# 
# Besides, you might also need tiles3.py from https://github.com/kamenbliznashki/sutton_barto/blob/master/tiles3.py to help you implement your code.

# In[ ]:


import gymnasium as gym
from math import floor, log
from itertools import zip_longest


# In[ ]:


"""
Tile Coding Software version 3.0beta
by Rich Sutton
based on a program created by Steph Schaeffer and others
External documentation and recommendations on the use of this code is available in the 
reinforcement learning textbook by Sutton and Barto, and on the web.
These need to be understood before this code is.

This software is for Python 3 or more.

This is an implementation of grid-style tile codings, based originally on
the UNH CMAC code (see http://www.ece.unh.edu/robots/cmac.htm), but by now highly changed. 
Here we provide a function, "tiles", that maps floating and integer
variables to a list of tiles, and a second function "tiles-wrap" that does the same while
wrapping some floats to provided widths (the lower wrap value is always 0).

The float variables will be gridded at unit intervals, so generalization
will be by approximately 1 in each direction, and any scaling will have 
to be done externally before calling tiles.

Num-tilings should be a power of 2, e.g., 16. To make the offsetting work properly, it should
also be greater than or equal to four times the number of floats.

The first argument is either an index hash table of a given size (created by (make-iht size)), 
an integer "size" (range of the indices from 0), or nil (for testing, indicating that the tile 
coordinates are to be returned without being converted to indices).
"""

basehash = hash

class IHT:
    "Structure to handle collisions"
    def __init__(self, sizeval):
        self.size = sizeval
        self.overfullCount = 0
        self.dictionary = {}

    def __str__(self):
        "Prepares a string for printing whenever this object is printed"
        return "Collision table:" + \
               " size:" + str(self.size) + \
               " overfullCount:" + str(self.overfullCount) + \
               " dictionary:" + str(len(self.dictionary)) + " items"

    def count (self):
        return len(self.dictionary)

    def fullp (self):
        return len(self.dictionary) >= self.size

    def getindex (self, obj, readonly=False):
        d = self.dictionary
        if obj in d: return d[obj]
        elif readonly: return None
        size = self.size
        count = self.count()
        if count >= size:
            if self.overfullCount==0: print('IHT full, starting to allow collisions')
            self.overfullCount += 1
            return basehash(obj) % self.size
        else:
            d[obj] = count
            return count

def hashcoords(coordinates, m, readonly=False):
    if type(m)==IHT: return m.getindex(tuple(coordinates), readonly)
    if type(m)==int: return basehash(tuple(coordinates)) % m
    if m==None: return coordinates

def tiles (ihtORsize, numtilings, floats, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q in qfloats:
            coords.append( (q + b) // numtilings )
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles

def tileswrap (ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=False):
    """returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
    qfloats = [floor(f*numtilings) for f in floats]
    Tiles = []
    for tiling in range(numtilings):
        tilingX2 = tiling*2
        coords = [tiling]
        b = tiling
        for q, width in zip_longest(qfloats, wrapwidths):
            c = (q + b%numtilings) // numtilings
            coords.append(c%width if width else c)
            b += tilingX2
        coords.extend(ints)
        Tiles.append(hashcoords(coords, ihtORsize, readonly))
    return Tiles


# In[ ]:


class MountainCarSARSAAgent(SemiGradientSARSAAgent):
    def __init__(self, env, info):
        self.env = env
        self.env.max_time_steps = env.spec.max_episode_steps

        self.episode_num = info['episode_num']
        self.alpha = info['alpha']
        self.gamma = info['gamma']
        self.epsilon = info['epsilon']
        self.params = info
        self.action_num = env.action_space.n
        self.state_space, self.state_num = self.create_state_aggregation()
        self.weights_fn = np.zeros(self.state_num)

    def create_state_aggregation(self):
        self.iht = IHT(4096)
        self.num_tilings = 8
        self.tile_size = [8 / (0.5 + 1.2), 8 / (0.07 + 0.07)]
        return self.iht, self.iht.size

    def _get_state_action_feature(self, state, action):
        tile_indices = tiles(self.iht, self.num_tilings, list(state), [action])
        feature_vector = np.zeros(self.iht.size)
        for idx in tile_indices:
            feature_vector[idx] = 1
        return feature_vector
    
    def epsilon_greedy_policy(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_values = [self.function_approximation(state, a) for a in range(self.action_num)]
            max_q = np.max(q_values)
            best_actions = [a for a, q in enumerate(q_values) if q == max_q]
            return np.random.choice(best_actions)


# In[ ]:


if __name__ == "__main__":
    # set the random seed
    np.random.seed(1234)
    random.seed(1234)

    params = {
        'episode_num': 500, # Set to 500 because of time constraints
        'alpha': 0.5 / 8,
        'gamma': 0.99,
        'epsilon': 0.1
    }

    run_trial = 10 # Set to 10 because of time constraints

    results_mountain_car = []
    for _ in range(run_trial):
        my_env = gym.make("MountainCar-v0")
        agent = MountainCarSARSAAgent(my_env, params)
        returns = agent.run()
        results_mountain_car.append(np.array(returns))


# In[ ]:


def plot_cost_to_go(agent, episode_num):
    positions = np.linspace(-1.2, 0.6, 50)
    velocities = np.linspace(-0.07, 0.07, 50)
    cost_to_go = np.zeros((len(positions), len(velocities)))

    for i, p in enumerate(positions):
        for j, v in enumerate(velocities):
            cost_to_go[i, j] = -max([agent.function_approximation([p, v], a) for a in range(3)])

    plt.imshow(cost_to_go, extent=[-1.2, 0.6, -0.07, 0.07], origin="lower", aspect='auto')
    plt.xlabel("Position")
    plt.ylabel("Velocity")
    plt.title(f"Cost-to-Go (Episode {episode_num})")
    plt.colorbar(label="Cost-to-Go")
    plt.show()


def plot_learning_curve_mountain_car(returns): 
    plt.figure(figsize=(10, 6))

    episodes = np.arange(1, len(returns[0]) + 1)
    average_steps = -np.array(returns)

    plt.plot(episodes, average_steps.mean(axis=0), color='blue')

    plt.title("Mountain Car Learning Curve") 
    plt.xlabel("Episode")
    plt.ylabel("Steps per episode")
    
    plt.show()


# In[ ]:


plot_cost_to_go(agent, params['episode_num'])
plot_learning_curve_mountain_car(results_mountain_car)

