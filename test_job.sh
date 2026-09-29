#!/bin/bash
#SBATCH --job-name=rclust_test
#SBATCH --output=rclust_test.out
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

echo "Hello from rclust!"
hostname
date
sleep 10
echo "Done."
