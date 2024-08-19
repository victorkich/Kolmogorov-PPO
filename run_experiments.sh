#!/bin/bash

# Check if seed is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <seed>"
  exit 1
fi

SEED=$1
# ENVS=("HalfCheetah-v4" "Walker2d-v4" "Hopper-v4" "InvertedPendulum-v4" "Swimmer-v4" "Pusher-v4" "Reacher-v4")
ENVS=("Walker2d-v4" "Hopper-v4" "InvertedPendulum-v4" "Swimmer-v4" "Pusher-v4" "Reacher-v4")

for ENV in "${ENVS[@]}"; do
  #python3 cleanrl/ppo_continuous_action.py --seed $SEED --env-id $ENV 
  python3 cleanrl/ppo_continuous_action_kan.py --seed $SEED --env-id $ENV --no-cuda
done