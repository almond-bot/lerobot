
#!/bin/bash

# Build uv sync command with extras based on arguments
EXTRAS="--extra almond"
[[ "$*" == *"--inference"* ]] && EXTRAS="$EXTRAS --extra feetech"
[[ "$*" == *"--train"* ]] && EXTRAS="$EXTRAS --extra pi0"

uv sync $EXTRAS

uv tool install pre-commit
pre-commit install

git config --global credential.helper store
git config --global user.email "workstation@almondbot.com"
git config --global user.name "Almond Workstation"

echo "Please enter your Hugging Face login token:"
read HUGGINGFACE_TOKEN

hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
