# ----- CREDITS: Chris Lu @ PureJaxRL -----
# https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/ppo.py


from typing import Sequence

import distrax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import tyro
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
from utils import PPO_Args, Transition


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


def make_train(arg):
    NUM_UPDATES = args.total_timesteps // args.num_steps // args.num_envs
    MINIBATCH_SIZE = args.num_envs * args.num_steps // args.num_minibatches
    env, env_params = gymnax.make(args.env_name)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0 - (count // (args.num_minibatches * args.update_epochs)) / NUM_UPDATES
        )
        return args.learning_rate * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).n, activation=args.activation
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if args.anneal_lr:
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(args.learning_rate, eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, args.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                """
                Steps the environment across ``num_envs``.
                Returns the updated runner state and observation.
                """
                train_state, env_state, last_obs, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                actions = pi.sample(seed=_rng)
                log_prob = pi.log_prob(actions)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, args.num_envs)
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, actions, env_params)
                transition = Transition(
                    done, actions, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, args.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = runner_state
            # get the last value estimate to initialize gae computation
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                """
                Compute the generalized advantage estimation of a trajectory batch.
                """

                def _get_advantages(gae_and_next_value, transition):
                    """
                    Iteratively computes the GAE starting from the last transition.
                    Uses `lax.scan` to carry the current (`gae`, `next_value`) tuple
                    while iterating through transitions.
                    """
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    # td-error
                    delta = reward + args.gamma * next_value * (1 - done) - value
                    # generalized advantage in recursive form
                    gae = delta + args.gamma * args.gae_lambda * (1 - done) * gae
                    return (gae, value), gae  # (carry_over), collected results

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    # gae is computed backwards as the advantage at time t
                    # depends on the estimated advantages of future timesteps
                    reverse=True,
                    # unrolls the loop body of the scan operation 16 iterations at a time
                    # enables the 128 steps (default value) to be completed in 8 iterations
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-args.clip_eps, args.clip_eps)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - args.clip_eps,
                                1.0 + args.clip_eps,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + args.vf_coef * value_loss
                            - args.ent_coef * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                # Batching and Shuffling
                batch_size = MINIBATCH_SIZE * args.num_minibatches
                assert (
                    batch_size == args.num_steps * args.num_envs
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                # Mini-batch Updates
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [args.num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            # Updating Training State and Metrics:
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, args.update_epochs
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            # Debugging mode
            if args.debug:

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * args.num_envs
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, NUM_UPDATES
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":

    args = tyro.cli(PPO_Args)
    print(
        f"Running {args.model_name} on {args.env_name} for {args.total_timesteps} steps..."
    )
    rng = jax.random.PRNGKey(args.seed)
    rngs = jax.random.split(rng, args.n_agents)
    train_vjit = jax.jit(jax.vmap(make_train(args)))
    outs = train_vjit(rngs)

    returns = outs["metrics"]["returned_episode_returns"]
    n_episodes = returns.shape[1]
    returns = outs["metrics"]["returned_episode_returns"].reshape(
        args.n_agents, n_episodes, -1
    )
    returns = pd.DataFrame(returns.transpose(1, 0, 2).reshape(n_episodes, -1))
    print(returns.shape)
    # pd.DataFrame(returns).to_csv(f"../../logs/{env_name}_parallel_ppo1.csv")
