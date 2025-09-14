import argparse
import subprocess

from env import HF_USER


def train_cmd(name: str, policy: str, steps: int, scratch: bool) -> list[str]:
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={HF_USER}/{name}",
        f"--policy.repo_id={HF_USER}/{policy}_{name}",
        f"--job_name={policy}_{name}",
        f"--output_dir=outputs/train/{policy}_{name}",
        "--policy.device=cuda",
        "--batch_size=256",
        f"--steps={steps}",
        "--wandb.enable=false",
    ]

    if policy == "act":
        cmd.append("--policy.type=act")
    elif policy == "smolvla":
        if scratch:
            cmd.append("--policy.type=smolvla")
        else:
            cmd.append("--policy.path=lerobot/smolvla_base")
    elif policy == "pi0":
        if scratch:
            cmd.append("--policy.type=pi0")
        else:
            cmd.append("--policy.path=lerobot/pi0")

    return cmd


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--policy", type=str, required=True)
parser.add_argument("--steps", type=int, required=True)
parser.add_argument("--scratch", action="store_true", default=False, required=False)
args = parser.parse_args()

cmd = train_cmd(**args.__dict__)
subprocess.run(cmd)
