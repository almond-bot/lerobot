import argparse
import subprocess

from env import HF_USER


def train_cmd(name: str, policy: str) -> list[str]:
    return [
        "lerobot-train",
        f"--dataset.repo_id={HF_USER}/{name}",
        f"--policy.type={policy}",
        f"--output_dir=outputs/train/{policy}_{name}",
        f"--job_name={policy}_{name}",
        "--policy.device=cuda",
        "--wandb.enable=false",
        f"--policy.repo_id={HF_USER}/{policy}_{name}",
    ]


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--policy", type=str, required=True)
args = parser.parse_args()

cmd = train_cmd(args.name, args.policy)
subprocess.run(cmd)
