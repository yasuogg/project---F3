#!/usr/bin/env bash
# Colab/Linux setup for RL-Augmented Vision Web Agent
set -e
apt-get -qq install -y xvfb libnss3 libgbm1 libasound2 libatk-bridge2.0-0 \
    libxkbcommon0 libxrandr2 libxcomposite1 libxdamage1
pip install -q -e . gradio
playwright install chromium
# start a virtual display in the background
Xvfb :99 -screen 0 1280x720x24 &
export DISPLAY=:99
echo "Setup complete. export DISPLAY=:99 in your shell."
