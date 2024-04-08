from dataclasses import dataclass

import tyro


@dataclass
class PPO_Args:
    model_name: str
    """name of the selected model"""
    seed: int = 0
    """experiment seed"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    log_results: bool = False
    """whether to save results in the logs folder"""

    wandb_project_name: str = "improved-gradient-steps"
    """the wandb project name"""
    wandb_entity: str = "rpegoud"
    """the entity (team) of wandb's project"""
    logging_dir: str = "."  # "$HOME/wandb"
    """the base directory for logging and wandb storage."""

    # Algorithm specific arguments
    env_name: str = "CartPole-v1"
    """environment to run"""
    total_timesteps: int = 5e4
    """total number of timesteps"""
    learning_rate: float = 2.5e-4
    """optimizer learning rate"""
    n_agents: int = 16
    """the number of parallel agents to train"""
    num_envs: int = 4
    """the number of parallel environments to collect transitions from"""
    num_steps: int = 128
    """the number of environment steps taken between gradient steps"""
    update_epochs: int = 4
    """the number of update steps in a single epoch"""
    num_minibatches: int = 4
    gamma: float = 0.99
    """discount factor"""
    gae_lambda: float = 0.95
    """the lambda exponent used in generalized advantage estimation"""
    clip_eps: float = 0.2
    """the threshold used to clip epsilon"""
    ent_coef: float = 0.5
    """entropy coefficient"""
    vf_coef: float = 0.5
    """value function coefficient"""
    max_grad_norm: float = 0.5
    """upper bound on the gradient norm"""
    alpha: float = 0.2
    """the alpha coefficient used in PPO1.c&d"""
    activation: str = "tanh"
    """the activation function used by the actor and critic networks"""
    anneal_lr: bool = True
    """whether to progressively reduce the learning rate or not"""
    debug: bool = False
    """toggle the debug mode"""


if __name__ == "__main__":
    args = tyro.cli(PPO_Args)
    print(args)
