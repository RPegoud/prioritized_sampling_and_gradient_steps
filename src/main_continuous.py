import datetime
import os
import time

import jax
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import tyro
import wandb
from continuous_algs import (
    base_ppo_continuous,
    parallel_ppo_1_continuous,
    parallel_ppo_1a_continuous,
    parallel_ppo_1b_continuous,
    parallel_ppo_1c_continuous,
    parallel_ppo_1d_continuous,
)
from utils import PPO_Continuous_Args, format_number

trainers = {
    "base_ppo_continuous": base_ppo_continuous,
    "parallel_ppo_1_continuous": parallel_ppo_1_continuous,
    "parallel_ppo_1a_continuous": parallel_ppo_1a_continuous,
    "parallel_ppo_1b_continuous": parallel_ppo_1b_continuous,
    "parallel_ppo_1c_continuous": parallel_ppo_1c_continuous,
    "parallel_ppo_1d_continuous": parallel_ppo_1d_continuous,
}

if __name__ == "__main__":
    args = tyro.cli(PPO_Continuous_Args)
    date = datetime.datetime.now()

    alpha = f"0_{int(args.alpha*10)}" if args.alpha != 0.2 else "0_2"

    id = f"{date.year}-{date.month}-{date.day}__{date.hour}_{date.minute}"
    run_name = f"igs__{args.trainer}{alpha}_{args.env_name}__{id}"
    print(f"Running {args.trainer} on {args.env_name} for {args.total_timesteps} steps")

    t = time.time()
    rng = jax.random.PRNGKey(args.seed)
    rngs = jax.random.split(rng, args.n_agents)
    train_vjit = jax.jit(jax.vmap(trainers[args.trainer](args)))
    outs = train_vjit(rngs)
    exec_time = time.gmtime(time.time() - t)
    print(f'Finished training in {time.strftime("%H:%M:%S", exec_time)}')

    # Metrics per episode
    avg_ep_returns = pd.Series(
        outs["metrics"]["returned_episode_returns"].mean(axis=(0, 2, 3))
    )
    std_ep_returns = pd.Series(
        outs["metrics"]["returned_episode_returns"].std(axis=(0, 2, 3))
    )
    n_episodes = avg_ep_returns.shape[0]

    path = f"logs/{args.env_name}"
    if not os.path.exists(path):
        os.makedirs(path)

    fmt_n_steps = format_number(args.total_timesteps)
    avg_ep_returns.to_csv(f"{path}/{run_name}_{fmt_n_steps}_avg_ep_returns.csv")
    std_ep_returns.to_csv(f"{path}/{run_name}_{fmt_n_steps}_std_ep_returns.csv")

    if args.log_results:
        hyperparameters = vars(args)
        html_table = "<table><tr><th>Parameter</th><th>Value</th></tr>"
        for key, value in hyperparameters.items():
            html_table += f"<tr><td>{key}</td><td>{value}</td></tr>"

        html_table += "</table>"
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            dir=args.logging_dir,
        )
        wandb.run.log_code(os.path.join(args.logging_dir, "/logs"))
        wandb.log({"hyperparameters": wandb.Html(html_table)})
        wandb.run.summary["trainer"] = args.trainer
        wandb.run.summary["total_timesteps"] = fmt_n_steps
        wandb.run.summary["n_episodes"] = n_episodes

        eps = np.arange(n_episodes)
        fig = go.Figure(
            [
                go.Scatter(
                    x=eps,
                    y=avg_ep_returns,
                    mode="lines",
                    name="Mean",
                ),
                go.Scatter(
                    x=eps,
                    y=avg_ep_returns + std_ep_returns,
                    line=dict(width=0),
                    showlegend=False,
                    mode="lines",
                    name="Upper Bound",
                    fill=None,
                ),
                go.Scatter(
                    x=eps,
                    y=avg_ep_returns - std_ep_returns,
                    line=dict(width=0),
                    mode="lines",
                    fill="tonexty",  # Fill area between y_upper and y_lower
                    fillcolor="rgba(0,191,255, 0.4)",
                    showlegend=False,
                    name="Lower Bound",
                ),
            ]
        )
        fig.update_layout(
            title=f"Returns over {n_episodes} episodes, averaged across {args.n_agents} agents, {args.trainer} - {args.env_name}",  # noqa: E501
            xaxis_title="Episodes",
            yaxis_title="Average return per episode",
            showlegend=False,
        )

        wandb.log({"Charts/average_ep_returns": wandb.Html(plotly.io.to_html(fig))})

        if not os.path.exists(f"logs/{args.env_name}"):
            os.makedirs(f"logs/{args.env_name}", exist_ok=True)

        artifact = wandb.Artifact(f"{run_name}_{fmt_n_steps}_artifacts", type="dataset")
        artifact.add_file(f"{path}/{run_name}_{fmt_n_steps}_avg_ep_returns.csv")
        artifact.add_file(f"{path}/{run_name}_{fmt_n_steps}_std_ep_returns.csv")
        wandb.log_artifact(artifact)
