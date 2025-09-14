import argparse
import queue
import subprocess
import threading
from datetime import datetime
from typing import Any

import requests
import utils
from env import HF_USER, SLACK_ENG_OPERATIONS_URL

logger = utils.create_logger(__name__)

# Message templates
SUCCESS_MESSAGE_TEMPLATE = """🎉 Training Completed Successfully!

📊 Training Details:
• Dataset: {name}
• Policy: {policy}
• Steps: {steps:,}
• From Scratch: {scratch}
• Started: {start_time}
• Finished: {end_time}
• Duration: {duration}
• Output Dir: outputs/train/{policy}_{name}
• Model Repo: {hf_user}/{policy}_{name}

✅ Training completed without errors!"""

FAILURE_MESSAGE_TEMPLATE = """❌ Training Failed!

📊 Training Details:
• Dataset: {name}
• Policy: {policy}
• Steps: {steps:,}
• From Scratch: {scratch}
• Started: {start_time}
• Failed: {end_time}
• Duration: {duration}

❌ Training failed with exit code: {return_code}
Error output: {error_output}"""

INTERRUPTED_MESSAGE_TEMPLATE = """⚠️ Training Interrupted!

📊 Training Details:
• Dataset: {name}
• Policy: {policy}
• Steps: {steps:,}
• Started: {start_time}
• Interrupted: {end_time}
• Duration: {duration}

⚠️ Training was manually interrupted (Ctrl+C)"""

ERROR_MESSAGE_TEMPLATE = """💥 Training Error!

📊 Training Details:
• Dataset: {name}
• Policy: {policy}
• Steps: {steps:,}
• Started: {start_time}
• Error: {end_time}
• Duration: {duration}

💥 Unexpected error: {error}"""


class SlackNotifier:
    """Handles Slack notifications for training events."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send_notification(self, message: str) -> None:
        """Send a notification to Slack using the webhook URL."""
        try:
            payload = {"text": message}
            response = requests.post(self.webhook_url, json=payload, timeout=30)
            response.raise_for_status()
            logger.info("Slack notification sent successfully!")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send Slack notification: {e}")

    def format_success_message(self, details: dict[str, Any]) -> str:
        """Format a success notification message."""
        return SUCCESS_MESSAGE_TEMPLATE.format(**details, hf_user=HF_USER)

    def format_failure_message(self, details: dict[str, Any], return_code: int, error_output: str) -> str:
        """Format a failure notification message."""
        return FAILURE_MESSAGE_TEMPLATE.format(
            **details,
            return_code=return_code,
            error_output=error_output[:500] if error_output else "No error output",
        )

    def format_interrupted_message(self, details: dict[str, Any]) -> str:
        """Format an interruption notification message."""
        return INTERRUPTED_MESSAGE_TEMPLATE.format(**details)

    def format_error_message(self, details: dict[str, Any], error: str) -> str:
        """Format an unexpected error notification message."""
        return ERROR_MESSAGE_TEMPLATE.format(**details, error=error[:500])


class TrainingRunner:
    """Manages the training process and notifications."""

    def __init__(self, notifier: SlackNotifier):
        self.notifier = notifier
        self.start_time = None
        self.end_time = None

    def build_train_command(self, name: str, policy: str, steps: int, scratch: bool) -> list[str]:
        """Build the training command based on parameters."""
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
        elif policy == "pi0fast":
            if scratch:
                cmd.append("--policy.type=pi0fast")
            else:
                cmd.append("--policy.path=lerobot/pi0fast_base")

        return cmd

    def create_training_details(self, args: argparse.Namespace) -> dict[str, Any]:
        """Create a dictionary with training details."""
        duration = self.end_time - self.start_time if self.end_time and self.start_time else None

        return {
            "name": args.name,
            "policy": args.policy,
            "steps": args.steps,
            "scratch": args.scratch,
            "start_time": self.start_time.strftime("%Y-%m-%d %H:%M:%S") if self.start_time else "Unknown",
            "end_time": self.end_time.strftime("%Y-%m-%d %H:%M:%S") if self.end_time else "Unknown",
            "duration": str(duration).split(".")[0] if duration else "Unknown",
        }

    def _stream_output(self, pipe, log_func, output_queue):
        """Stream output from a pipe to the logger and collect it in a queue."""
        try:
            for line in iter(pipe.readline, ""):
                if line:
                    line = line.rstrip()
                    log_func(line)
                    output_queue.put(line)
            pipe.close()
        except Exception as e:
            logger.error(f"Error streaming output: {e}")

    def run_training(self, args: argparse.Namespace) -> None:
        """Execute the training process with proper error handling and notifications."""
        cmd = self.build_train_command(args.name, args.policy, args.steps, args.scratch)
        self.start_time = datetime.now()

        logger.info(f"Starting training at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Running command: {' '.join(cmd)}")

        # Queues to collect output for error reporting
        stdout_queue = queue.Queue()
        stderr_queue = queue.Queue()

        try:
            # Start the process with pipes for real-time streaming
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # Start threads to stream stdout and stderr
            stdout_thread = threading.Thread(
                target=self._stream_output, args=(process.stdout, logger.info, stdout_queue)
            )
            stderr_thread = threading.Thread(
                target=self._stream_output, args=(process.stderr, logger.error, stderr_queue)
            )

            stdout_thread.start()
            stderr_thread.start()

            # Wait for the process to complete
            return_code = process.wait()

            # Wait for output threads to finish
            stdout_thread.join()
            stderr_thread.join()

            self.end_time = datetime.now()
            details = self.create_training_details(args)

            # Collect stderr output for error reporting
            stderr_output = ""
            while not stderr_queue.empty():
                try:
                    stderr_output += stderr_queue.get_nowait() + "\n"
                except queue.Empty:
                    break

            if return_code == 0:
                message = self.notifier.format_success_message(details)
                self.notifier.send_notification(message)
                exit(0)
            else:
                message = self.notifier.format_failure_message(details, return_code, stderr_output)
                self.notifier.send_notification(message)
                exit(return_code)

        except KeyboardInterrupt:
            # Terminate the process if it's still running
            if "process" in locals() and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            self._handle_interruption(args)
        except Exception as e:
            self._handle_unexpected_error(args, str(e))

    def _handle_interruption(self, args: argparse.Namespace) -> None:
        """Handle training interruption (Ctrl+C)."""
        self.end_time = datetime.now()
        details = self.create_training_details(args)
        message = self.notifier.format_interrupted_message(details)
        self.notifier.send_notification(message)
        logger.info("\nTraining interrupted by user")
        exit(1)

    def _handle_unexpected_error(self, args: argparse.Namespace, error: str) -> None:
        """Handle unexpected errors during training."""
        self.end_time = datetime.now()
        details = self.create_training_details(args)
        message = self.notifier.format_error_message(details, error)
        self.notifier.send_notification(message)
        logger.error(f"Training failed with error: {error}")
        exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a robot learning policy with Slack notifications")
    parser.add_argument("--name", type=str, required=True, help="Dataset name")
    parser.add_argument("--policy", type=str, required=True, help="Policy type (act, smolvla, pi0, pi0fast)")
    parser.add_argument("--steps", type=int, required=True, help="Number of training steps")
    parser.add_argument(
        "--scratch",
        action="store_true",
        default=False,
        help="Train from scratch instead of using pretrained model",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the training script."""
    args = parse_arguments()

    # Initialize components
    notifier = SlackNotifier(SLACK_ENG_OPERATIONS_URL)
    trainer = TrainingRunner(notifier)

    # Run training
    trainer.run_training(args)


if __name__ == "__main__":
    main()
