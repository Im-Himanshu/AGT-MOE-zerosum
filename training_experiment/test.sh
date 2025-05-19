#!/bin/bash

# Simulated job 1
{
  echo "Job 1 started at $(date)"
  python run_experiment.py
  echo "Job 1 ended at $(date)"
} &

sleep 30  # Delay before launching job 2

# Simulated job 2
{
  echo "Job 2 started at $(date)"
  python run_experiment.py
  echo "Job 2 ended at $(date)"
} &

sleep 30  # Delay before launching job 3

# Simulated job 3
{
  echo "Job 3 started at $(date)"
  python run_experiment.py
  echo "Job 3 ended at $(date)"
} &

{
  while true; do
    echo "=== nvidia-smi at $(date) ===" >> gpu_log.txt
    nvidia-smi >> gpu_log.txt
    echo "" >> gpu_log.txt
    sleep 360
  done
} &


wait
echo "All jobs completed at $(date)"
