import train
import wandb
import yaml
import argparse


def parse_args():
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description='Hyparam sweep')

    parser.add_argument('--config_path', type=str, help='Path to the configuration file',
                        default="configs/default.yaml", required=False)
    parser.add_argument("--sweep_config_path", type=str, help='Path to the sweep configuration file',
                        default="configs/sweep_config.yaml", required=False)
    parser.add_argument("--sweep_id", type=str, help='Sweep ID',
                        default=None, required=False)
    parser.add_argument("--saving_root_dir", type=str, help='Path to where to save project files',
                        default="./", required=False)
    parser.add_argument('--project', type=str, help='Wandb project name',
                        default="DiffLand", required=False)
    parser.add_argument("--desc", type=str, help="Description of the run", default="", required=False)
    parser.add_argument("--runs", type=int, help="Number of runs", default=3, required=False)
    # Parse the command-line arguments
    args = parser.parse_args()
    return args


def main():
    """
    parse args
    if sweep_id is not None, then run the agent
        wandb.agent(sweep_id, function=train.main, count=4)
    else create a sweep
        load the sweep configuration from the file
        create a sweep
        output the sweep id
    """
    args = parse_args()
    if args.sweep_id is not None:
        f = lambda: train.main(args)
        # f = lambda: train_ensemble.main(args)
        wandb.agent(args.sweep_id, function=f, count=args.runs, project=args.project)
        return
    with open(args.sweep_config_path, "r") as f:
        sweep_cfg = yaml.safe_load(f)

    sweep_id = wandb.sweep(sweep=sweep_cfg, project=args.project)
    print(f"Sweep ID: {sweep_id}")


if __name__ == "__main__":
    main()
    # import os
    #
    # os.environ["WANDB_START_METHOD"] = "thread"
    # wandb.agent("zno7093d", function=init_train_test, count=2)
