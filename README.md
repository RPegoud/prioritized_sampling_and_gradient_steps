# ***Gradient Experience Replay and Improved Gradient Steps***

## ***Installation:***

```bash
pip install poetry
poetry install
poetry shell
python .\src\{model_to_run}.py --{args values}
```

## ***Running an experiment:***

This repository uses [***tyro***](https://brentyi.github.io/tyro/) as a CLI interface to run experiments, default values and definitions are stored in `utils/arg_parser.py`.

```bash
python .\src\base_ppo.py --model-name ppo --total-timesteps 10000
```

## Command Line Arguments

| Argument                  | Type   | Default           | Description                                                                                       |
|---------------------------|--------|-------------------|---------------------------------------------------------------------------------------------------|
| `--model-name`            | STR    | *Required*        | Name of the selected model.                                                                       |
| `--exp-name`              | STR    | `arg_parser`      | Experiment name.                                                                                  |
| `--seed`                  | INT    | `0`               | Experiment seed.                                                                                  |
| `--save-model`/`--no-save-model` | BOOL      | `False`           | Whether to save model into the `runs/{run_name}` folder.                                          |
| `--env-name`              | STR    | `CartPole-v1`     | Environment to run.                                                                               |
| `--total-timesteps`       | INT    | `50000`           | Total number of timesteps.                                                                        |
| `--learning-rate`         | FLOAT  | `0.00025`         | Optimizer learning rate.                                                                          |
| `--n-agents`              | INT    | `16`              | The number of parallel agents to train.                                                          |
| `--num-envs`              | INT    | `4`               | The number of parallel environments to collect transitions from.                                  |
| `--num-steps`             | INT    | `128`             | The number of environment steps taken between gradient steps.                                     |
| `--update-epochs`         | INT    | `4`               | The number of update steps in a single epoch.                                                    |
| `--num-minibatches`       | INT    | `4`               | Algorithm specific arguments.                                                                     |
| `--gamma`                 | FLOAT  | `0.99`            | Discount factor.                                                                                  |
| `--gae-lambda`            | FLOAT  | `0.95`            | The lambda exponent used in generalized advantage estimation.                                     |
| `--clip-eps`              | FLOAT  | `0.2`             | The threshold used to clip epsilon.                                                               |
| `--ent-coef`              | FLOAT  | `0.5`             | Entropy coefficient.                                                                              |
| `--vf-coef`               | FLOAT  | `0.5`             | Value function coefficient.                                                                       |
| `--max-grad-norm`         | FLOAT  | `0.5`             | Upper bound on the gradient norm.                                                                 |
| `--activation`            | STR    | `tanh`            | The activation function used by the actor and critic networks.                                    |
| `--anneal-lr`/`--no-anneal-lr` | BOOL      | `True`            | Whether to progressively reduce the learning rate or not.                                         |
| `--debug`/`--no-debug`    | BOOL      | `False`           | Toggle the debug mode.                                                                            |
