import numpy as np
import torch
from torch import nn, Tensor
from typing import Optional, Callable

import utils

EPSILON = 1e-10

def get_returns(rollout_buffer: utils.RolloutBuffer, discount_factor=0.95):
    """ Calculate the returns, given the rollout buffer. The rollout buffer should contain some number of rollouts,
        concatenated together.
        Inputs:
            rollout_buffer: utils.RolloutBuffer
                Contains the rollout buffer. Note that this contains multiple rollouts, so you need to re-initialize
                the reward for terminal states.
                That is, if rollout_buffer.terminateds[i] is True, then rollout_buffer.rewards[i] is the terminal reward
                for a rollout, and index i+1 starts the recording of a new rollout.
            discount_factor: float
                Gamma. Multiplied to the reward geometrically (see notebook)
        Outputs:
            torch.Tensor[torch.float32], timesteps x 1
                Returns for the entire set of rollouts
    """
    rewards = torch.tensor(rollout_buffer.rewards, dtype=torch.float32)
    terminateds = torch.tensor(rollout_buffer.terminateds, dtype=torch.bool)
    returns = torch.zeros_like(rewards)
    
    # Start from the last timestep and move backwards
    current_return = 0.0
    for i in reversed(range(len(rewards))):
        if terminateds[i]:  # Reset the return upon encountering a terminal state
            current_return = 0.0
        current_return = rewards[i] + discount_factor * current_return
        returns[i] = current_return
    
    return returns.unsqueeze(1)  # Reshape returns to have shape (timesteps, 1)


def get_advantages(value_net: nn.Module,
                   observations: torch.Tensor,
                   returns: torch.Tensor):
    """ Get the advantages for the given rollout buffer.
        Inputs:
            value_net: nn.Module
                value net (critic) to get the training loss for
            observations: torch.Tensor[torch.float32], batch x obs_dim
                Contains the observations
            returns: torch.Tensor[torch.float32], batch x 1.
                Future returns associated with the given observations
        Outputs:
            torch.Tensor[torch.float32], singleton
                Value network loss for the given returns
    """
    # YOUR CODE HERE
    # Remember to use torch.no_grad!
    # You should calculate the advantage, then standardize it (subtract out mean, then divide by standard deviation
    # plus epsilon.) Use 1e-10 for epsilon (defined as EPSILON at top of file). Epsilon is solely there to prevent
    # divide-by-zero errors.
    
    """ Calculate the advantages as the difference between returns and predicted values, then standardize. """
    with torch.no_grad():  # No need to track gradients for this operation
        predicted_values = value_net(observations)  # Get the predicted values from the value network
    advantages = returns - predicted_values  # Calculate raw advantages

    # Standardize the advantages (improves training stability)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)  # Add epsilon to avoid division by zero
    return advantages

def get_value_net_loss(value_net: nn.Module,
                       observation: Tensor,
                       returns: Tensor,
                       **kwargs):
    """ Get the training loss for the value network V(s_t)
        Inputs:
            value_net: nn.Module
                value net (critic) to get the training loss for
            observation: torch.Tensor[torch.float32], batch x obs_dim
                Observation from the environment, given to the policy
            returns: torch.Tensor[torch.float32], batch x 1.
                Future returns associated with the given observations
        Outputs:
            torch.Tensor[torch.float32], singleton
                Value network loss for the given returns
    """
    # YOUR CODE HERE
    
    """ Calculate the MSE loss for the value network against given returns. """
    predicted_values = value_net(observation)  # Get the predicted values from the value network
    loss = torch.nn.functional.mse_loss(predicted_values, returns)  # Compute MSE loss
    return loss

def get_vanilla_policy_gradient_loss(policy: nn.Module,
                                observation: Tensor,
                                action: Tensor,
                                return_or_advantage: Tensor,
                                **kwargs):
    """ Get the return-based policy gradient loss for a minibatch. Each batch element represents a different timestep
        Inputs:
            policy: nn.Module
                Policy to get the policy gradient loss for
            observation: torch.Tensor[torch.float32], batch x obs_dim
                Observation from the environment, given to the policy
            action: torch.Tensor[torch.float32], batch x 1
                Action taken in the rollout for the associated observation
            return_or_advantage: torch.Tensor[torch.float32], batch x 1.
                Future return, if taking associated action, or separately computed advantage
        Outputs:
            torch.Tensor[torch.float32], singleton
                Vanilla policy gradient loss for the given return or advantage
    """
    # Generate log probabilities from the policy network
    log_probs = policy(observation)  # Shape: (batch_size, action_space_size)
    
    # Extract the log probabilities corresponding to the taken actions
    # Actions are assumed to be provided as (batch_size, 1) and contain indices of taken actions
    action_log_probs = log_probs.gather(1, action).squeeze(-1)  # Remove last dimension to match return shape
    
    # Calculate the policy gradient loss
    loss = -(action_log_probs * return_or_advantage.squeeze()).mean()  # Taking mean for average over batch
    
    return loss

def collect_rollouts(env: utils.EnvInterface,
                     policy: nn.Module,
                     num_rollouts: int,
                     base_rollout: int = 0, # for logging only
                     num_total_rollouts: int = None, # for logging only
                     seed = None):
    num_total_rollouts = num_rollouts if num_total_rollouts is None else num_total_rollouts
    rollout_buffer = utils.RolloutBuffer()
    final_reward_mean = []
    for i in range(num_rollouts):
        print(f"Runing rollout {base_rollout + i}/{num_total_rollouts}", end="\r")
        obs = env.reset()
        terminated = False
        policy.eval() # Put the policy in eval mode
        
        total_reward = 0
        
        while not terminated:
            # YOUR CODE HERE.
            # Steps:
            # 1) Evaluate the policy to get logits. Remember to use torch.no_grad here!
            # 2) Sample an action based on the logits. Use the provided line for this! The seed is important for grading.
            #       Provided line: action = utils.distribution_sample(logits, seed=seed)
            # 3) Execute the action on the environment with env.step
            # 4) Save the step to the rollout buffer with rollout_buffer.add. Provide this method with
            #       the action taken, the network logit outputs, the observation, whether the environment terminated,
            #       and the reward
            # 5) Think about what the observation should be for the next step.
            # Note: final_reward_mean is not required for grading, but not having breaks the notebook. Variable reward
            # should be the last reward of the rollout for it to work.
            
            with torch.no_grad():  # Ensure no gradients are computed during the rollout
                logits = policy(obs)
            action = utils.distribution_sample(logits, seed=seed)  # Sample an action based on the policy's output
            obs, terminated, reward = env.step(action)  # Execute the action and receive the next state and reward
            total_reward += reward  # Accumulate the reward

            # Store the results of this step into the rollout buffer
            rollout_buffer.add(action=action, logits=logits, observation=obs, terminated=terminated, reward=reward)

            
            # END YOUR CODE
        final_reward_mean.append(total_reward)
    policy.train() # Put the policy back in train mode
    rollout_buffer.finalize()

    return rollout_buffer, np.mean(final_reward_mean) if final_reward_mean else 0

def train_policy_gradient(env: utils.EnvInterface,
                          policy: nn.Module,
                          optimizer: torch.optim.Optimizer,
                          get_policy_gradient_loss: Callable,
                          get_returns: Callable,
                          lr_sched: torch.optim.lr_scheduler.LRScheduler = None,
                          rollouts: int = 1,
                          rollouts_before_training: int = 1,
                          training_epochs_per_rollout: int = 1,
                          minibatch_size: int = 64,
                          ppo_clip: float = 0.2,
                          critic_loss_multiplier: float = 0.0,
                          value_net: Optional[nn.Module] = None,
                          get_advantages: Optional[Callable] = None,
                          get_value_net_loss: Optional[Callable] = lambda **kwargs: torch.tensor([0.]),
                          rollout_seed=None):
    """ Trains policy gradient for the given number of rollouts.
        Inputs:
            env: EnvInterface
                Environment. Has methods reset() and step(). reset outputs observation only, while step outputs
                observation and reward alongside whether the environment has terminated.
                All observations are returned as 1 x obs_dim torch tensors, and actions should be 1 x action_dim
                torch tensors.
            policy: nn.Module
                policy to train
            optimizer:
                optimizer for the policy
            get_policy_gradient_loss: Callable
                Function called on a collection of kwargs (provided) that calculates the policy gradient
            get_returns: Callable
                Function called on a RolloutBuffer that gives the total return at each timestep
            lr_sched:
                lr scheduler for the optimizer
            rollouts: int
                Number of rollouts (reset call to step returning a termination) to run
            rollouts_before_training: int
                Number of rollouts to collect before training
            training_epochs_per_rollout: int
                Number of training epochs to run for one rollout
            minibatch_size: int
                Minibatch size for training
            ppo_clip: float
                clip constant passed through to the ppo loss
            value_net: nn.Module
                Value estimation network, optional.
            get_advantages: Callable
                Function called on the value net, a RolloutBuffer, and a list of returns that gives the advantage
                at each timestep
            get_value_net_loss: Callable
                Function called on a collection of kwargs (provided) that calculates the policy gradient
            rollout_seed
                Passthrough to collect_rollouts. For grading purposes.
    """
    r = 0
    losses_actor = []
    losses_critic = []
    final_rewards = []
    lr = []
    while r < rollouts:
        rollout_buffer, final_reward = collect_rollouts(env, policy, rollouts_before_training, r, rollouts, seed=rollout_seed)
        r += rollouts_before_training
        returns = get_returns(rollout_buffer)
        final_rewards.append(final_reward)
        if get_advantages:
            advantages = get_advantages(value_net, rollout_buffer.observations, returns)
        for _ in range(training_epochs_per_rollout):
            idxr_base = np.arange(rollout_buffer.final_size)
            np.random.shuffle(idxr_base)
            for batch_start in range(0, rollout_buffer.final_size, minibatch_size):
                batch_stop = min(batch_start + minibatch_size, rollout_buffer.final_size)
                # YOUR CODE HERE
                # Remember to fill out collect_rollouts() as well!
                # Fill out the items in the policy gradient kwargs dict. Slice the lists in the rollout buffer
                # and/or returns with the batch. return_or_advantage should be advantages if get_advantages is provided
                # (i.e. not None), or returns otherwise.
                # If you have not reached the Advantage section yet, do not worry about this, and just provide correctly
                # sliced returns.
                # Everything should be a torch tensor, as specified by the inputs to get_PPO_policy_gradient_loss and
                # get_vanilla_policy_gradient_loss
                
                indices = idxr_base[batch_start:batch_stop]

                batch_observations = rollout_buffer.observations[indices]
                batch_actions = rollout_buffer.actions[indices]
                batch_old_logits = rollout_buffer.old_logits[indices]
                batch_returns = returns[indices]
                batch_advantages = advantages[indices] if get_advantages else batch_returns  # Use advantages if available

                
                policy_gradient_kwargs = dict(
                    policy=policy,                 # The policy network you are training
                    value_net=value_net,           # The value network, if you are using one (for algorithms like PPO)
                    critic=value_net,              # This is often the same as the value_net; used interchangeably in some contexts
                    observation=batch_observations, # Batched observations from the rollout buffer
                    old_logits=batch_old_logits,    # Batched logits from when the actions were taken
                    action=batch_actions,           # Batched actions taken during the rollouts
                    return_or_advantage=batch_advantages, # Batched advantages if available, otherwise batched returns
                    returns=batch_returns,          # Batched returns, needed for training the value network
                    ppo_clip=ppo_clip               # The PPO clip parameter, important for PPO loss calculation
                )
                
                
                if value_net:  # If there's a value network, include it and possibly other needed items
                    policy_gradient_kwargs.update({
                        'value_net': value_net,
                        'critic': value_net  # Sometimes 'critic' is used instead of 'value_net'
                    })


                # END YOUR CODE
                optimizer.zero_grad()
                loss_actor = get_policy_gradient_loss(**policy_gradient_kwargs)
                loss_critic = get_value_net_loss(**policy_gradient_kwargs)
                loss = loss_actor + loss_critic * critic_loss_multiplier
                loss.backward()
                optimizer.step()
                losses_actor.append(loss_actor.detach().numpy())
                losses_critic.append(loss_critic.detach().numpy())                
                if lr_sched is not None:
                    lr_sched.step()
                    lr.append(optimizer.param_groups[0]['lr'])
    return losses_actor, losses_critic, final_rewards, lr

def get_PPO_policy_gradient_loss (policy: nn.Module,
                            observation: Tensor,
                            old_logits: Tensor,
                            action: Tensor,
                            return_or_advantage: Tensor,
                            ppo_clip = 0.2,
                            **kwargs):
    """ Get the return-based policy gradient loss for a minibatch. Each batch element represents a different timestep
        Inputs:
            policy: nn.Module.
                Policy to get the policy gradient loss for
            observation: torch.Tensor[torch.float32], batch x obs_dim
                Observation from the environment, given to the policy
            old_logits: torch.Tensor[torch.float32], batch x action_dim
                Logits output by the policy during runtime
            action: torch.Tensor[torch.float32], batch x 1
                Action taken in the rollout for the associated observation
            return_or_advantage: torch.Tensor[torch.float32], batch x 1.
                Separately computed advantage for the associated action
            ppo_clip: float
                PPO clipping epsilon on the probability ratio
    """
    # YOUR CODE HERE
    
    """ Calculate the Proximal Policy Optimization (PPO) policy gradient loss. """
    # Calculate the current logits and the log probabilities for the actions taken
    current_logits = policy(observation)
    log_probs = torch.nn.functional.log_softmax(current_logits, dim=-1)
    actions_log_probs = log_probs.gather(1, action.long())

    # Calculate the log probabilities from the old logits (old policy)
    old_log_probs = torch.nn.functional.log_softmax(old_logits, dim=-1)
    old_actions_log_probs = old_log_probs.gather(1, action.long())

    # Compute the ratio of the probabilities of the current policy to the old policy
    ratios = torch.exp(actions_log_probs - old_actions_log_probs)

    # Calculate the clipped part of the objective
    clipped_ratios = torch.clamp(ratios, 1 - ppo_clip, 1 + ppo_clip)
    clipped_objective = clipped_ratios * return_or_advantage

    # Combine the unclipped and clipped objectives using the minimum
    objective = torch.min(ratios * return_or_advantage, clipped_objective)

    # The loss is the negative mean of the combined objective, as we perform gradient ascent on the policy objective
    loss = -torch.mean(objective)
    return loss