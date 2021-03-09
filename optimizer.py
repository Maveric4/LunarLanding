import numpy as np


class Adam():
    # Work Required: Yes. Fill in the initialization for self.m and self.v (~4 Lines).
    def __init__(self, layer_sizes,
                 optimizer_info):
        self.layer_sizes = layer_sizes

        # Specify Adam algorithm's hyper parameters
        self.step_size = optimizer_info.get("step_size")
        self.beta_m = optimizer_info.get("beta_m")
        self.beta_v = optimizer_info.get("beta_v")
        self.epsilon = optimizer_info.get("epsilon")

        # Initialize Adam algorithm's m and v
        self.m = [dict() for i in range(1, len(self.layer_sizes))]
        self.v = [dict() for i in range(1, len(self.layer_sizes))]

        for i in range(0, len(self.layer_sizes) - 1):
            ### START CODE HERE (~4 Lines)
            # Hint: The initialization for m and v should look very much like the initializations of the weights
            # except for the fact that initialization here is to zeroes (see description above.)
            self.m[i]["W"] = np.zeros((self.layer_sizes[i], self.layer_sizes[i + 1]))
            self.m[i]["b"] = np.zeros((1, self.layer_sizes[i + 1]))
            self.v[i]["W"] = np.zeros((self.layer_sizes[i], self.layer_sizes[i + 1]))
            self.v[i]["b"] = np.zeros((1, self.layer_sizes[i + 1]))
            ### END CODE HERE

        # Notice that to calculate m_hat and v_hat, we use powers of beta_m and beta_v to
        # the time step t. We can calculate these powers using an incremental product. At initialization then,
        # beta_m_product and beta_v_product should be ...? (Note that timesteps start at 1 and if we were to
        # start from 0, the denominator would be 0.)
        self.beta_m_product = self.beta_m
        self.beta_v_product = self.beta_v

    # Work Required: Yes. Fill in the weight updates (~5-7 lines).
    def update_weights(self, weights, td_errors_times_gradients):
        """
        Args:
            weights (Array of dictionaries): The weights of the neural network.
            td_errors_times_gradients (Array of dictionaries): The gradient of the
            action-values with respect to the network's weights times the TD-error
        Returns:
            The updated weights (Array of dictionaries).
        """
        for i in range(len(weights)):
            for param in weights[i].keys():
                ### START CODE HERE (~5-7 Lines)
                # Hint: Follow the equations above. First, you should update m and v and then compute
                # m_hat and v_hat. Finally, compute how much the weights should be incremented by.
                self.m[i][param] = self.beta_m * self.m[i][param] + (1 - self.beta_m) * td_errors_times_gradients[i][
                    param]
                self.v[i][param] = self.beta_v * self.v[i][param] + (1 - self.beta_v) * td_errors_times_gradients[i][
                    param] ** 2
                m_hat = self.m[i][param] / (1 - self.beta_m_product)
                v_hat = self.v[i][param] / (1 - self.beta_v_product)
                weight_update = self.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)
                ### END CODE HERE

                weights[i][param] = weights[i][param] + weight_update
        # Notice that to calculate m_hat and v_hat, we use powers of beta_m and beta_v to
        ### update self.beta_m_product and self.beta_v_product
        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v

        return weights


def softmax(action_values, tau=1.0):
    """
    Args:
        action_values (Numpy array): A 2D array of shape (batch_size, num_actions).
                       The action-values computed by an action-value network.
        tau (float): The temperature parameter scalar.
    Returns:
        A 2D array of shape (batch_size, num_actions). Where each column is a probability distribution over
        the actions representing the policy.
    """
    ### START CODE HERE (~2 Lines)
    # Compute the preferences by dividing the action-values by the temperature parameter tau
    preferences = action_values / tau
    # Compute the maximum preference across the actions
    max_preference = np.max(preferences, axis=1)
    ### END CODE HERE

    # Reshape max_preference array which has shape [Batch,] to [Batch, 1]. This allows NumPy broadcasting
    # when subtracting the maximum preference from the preference of each action.
    reshaped_max_preference = max_preference.reshape((-1, 1))

    ### START CODE HERE (~2 Lines)
    # Compute the numerator, i.e., the exponential of the preference - the max preference.
    exp_preferences = np.exp(preferences - reshaped_max_preference)
    # Compute the denominator, i.e., the sum over the numerator along the actions axis.
    sum_of_exp_preferences = np.sum(exp_preferences, axis=1)
    ### END CODE HERE

    # Reshape sum_of_exp_preferences array which has shape [Batch,] to [Batch, 1] to  allow for NumPy broadcasting
    # when dividing the numerator by the denominator.
    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1, 1))

    ### START CODE HERE (~1 Lines)
    # Compute the action probabilities according to the equation in the previous cell.
    action_probs = exp_preferences / reshaped_sum_of_exp_preferences
    ### END CODE HERE

    # squeeze() removes any singleton dimensions. It is used here because this function is used in the
    # agent policy when selecting an action (for which the batch dimension is 1.) As np.random.choice is used in
    # the agent policy and it expects 1D arrays, we need to remove this singleton batch dimension.
    action_probs = action_probs.squeeze()
    return action_probs


def get_td_error(states, next_states, actions, rewards, discount, terminals, network, current_q, tau):
    """
    Args:
        states (Numpy array): The batch of states with the shape (batch_size, state_dim).
        next_states (Numpy array): The batch of next states with the shape (batch_size, state_dim).
        actions (Numpy array): The batch of actions with the shape (batch_size,).
        rewards (Numpy array): The batch of rewards with the shape (batch_size,).
        discount (float): The discount factor.
        terminals (Numpy array): The batch of terminals with the shape (batch_size,).
        network (ActionValueNetwork): The latest state of the network that is getting replay updates.
        current_q (ActionValueNetwork): The fixed network used for computing the targets,
                                        and particularly, the action-values at the next-states.
    Returns:
        The TD errors (Numpy array) for actions taken, of shape (batch_size,)
    """

    # Note: Here network is the latest state of the network that is getting replay updates. In other words,
    # the network represents Q_{t+1}^{i} whereas current_q represents Q_t, the fixed network used for computing the
    # targets, and particularly, the action-values at the next-states.

    # Compute action values at next states using current_q network
    # Note that q_next_mat is a 2D array of shape (batch_size, num_actions)

    ### START CODE HERE (~1 Line)
    q_next_mat = current_q.get_action_values(next_states)
    ### END CODE HERE

    # Compute policy at next state by passing the action-values in q_next_mat to softmax()
    # Note that probs_mat is a 2D array of shape (batch_size, num_actions)

    ### START CODE HERE (~1 Line)
    probs_mat = softmax(q_next_mat, tau)
    ### END CODE HERE

    # Compute the estimate of the next state value, v_next_vec.
    # Hint: sum the action-values for the next_states weighted by the policy, probs_mat. Then, multiply by
    # (1 - terminals) to make sure v_next_vec is zero for terminal next states.
    # Note that v_next_vec is a 1D array of shape (batch_size,)

    ### START CODE HERE (~3 Lines)
    v_next_vec = np.sum(q_next_mat * probs_mat, axis=1) * (1 - terminals)
    ### END CODE HERE

    # Compute Expected Sarsa target
    # Note that target_vec is a 1D array of shape (batch_size,)

    ### START CODE HERE (~1 Line)
    target_vec = rewards + discount * v_next_vec
    ### END CODE HERE

    # Compute action values at the current states for all actions using network
    # Note that q_mat is a 2D array of shape (batch_size, num_actions)

    ### START CODE HERE (~1 Line)
    q_mat = network.get_action_values(states)
    ### END CODE HERE

    # Batch Indices is an array from 0 to the batch size - 1.
    batch_indices = np.arange(q_mat.shape[0])

    # Compute q_vec by selecting q(s, a) from q_mat for taken actions
    # Use batch_indices as the index for the first dimension of q_mat
    # Note that q_vec is a 1D array of shape (batch_size)

    ### START CODE HERE (~1 Line)
    q_vec = q_mat[batch_indices, actions]
    ### END CODE HERE

    # Compute TD errors for actions taken
    # Note that delta_vec is a 1D array of shape (batch_size)

    ### START CODE HERE (~1 Line)
    delta_vec = target_vec - q_vec
    ### END CODE HERE

    return delta_vec


# Now that you implemented the `get_td_error()` function, you can use it to implement the `optimize_network()` function. In this function, you will:
# - get the TD-errors vector from `get_td_error()`,
# - make the TD-errors into a matrix using zeroes for actions not taken in the transitions,
# - pass the TD-errors matrix to the `get_TD_update()` function of network to calculate the gradients times TD errors, and,
# - perform an ADAM optimizer step.

# In[ ]:


### Work Required: Yes. Fill in code in optimize_network (~2 Lines).
def optimize_network(experiences, discount, optimizer, network, current_q, tau):
    """
    Args:
        experiences (Numpy array): The batch of experiences including the states, actions,
                                   rewards, terminals, and next_states.
        discount (float): The discount factor.
        network (ActionValueNetwork): The latest state of the network that is getting replay updates.
        current_q (ActionValueNetwork): The fixed network used for computing the targets,
                                        and particularly, the action-values at the next-states.
    """

    # Get states, action, rewards, terminals, and next_states from experiences
    states, actions, rewards, terminals, next_states = map(list, zip(*experiences))
    states = np.concatenate(states)
    next_states = np.concatenate(next_states)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    batch_size = states.shape[0]

    # Compute TD error using the get_td_error function
    # Note that q_vec is a 1D array of shape (batch_size)
    delta_vec = get_td_error(states, next_states, actions, rewards, discount, terminals, network, current_q, tau)

    # Batch Indices is an array from 0 to the batch_size - 1.
    batch_indices = np.arange(batch_size)

    # Make a td error matrix of shape (batch_size, num_actions)
    # delta_mat has non-zero value only for actions taken
    delta_mat = np.zeros((batch_size, network.num_actions))
    delta_mat[batch_indices, actions] = delta_vec

    # Pass delta_mat to compute the TD errors times the gradients of the network's weights from back-propagation

    ### START CODE HERE
    td_update = network.get_TD_update(states, delta_mat)
    ### END CODE HERE

    # Pass network.get_weights and the td_update to the optimizer to get updated weights
    ### START CODE HERE
    weights = optimizer.update_weights(network.get_weights(), td_update)
    ### END CODE HERE

    network.set_weights(weights)


