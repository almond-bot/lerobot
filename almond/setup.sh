#!/bin/bash

uv sync --extra almond,feetech,async,smolvla,pi0

uv tool install pre-commit
pre-commit install

git config --global credential.helper store
git config --global user.email "workstation@almondbot.com"
git config --global user.name "Almond Workstation"

echo "Please enter your Hugging Face login token:"
read HUGGINGFACE_TOKEN

hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
