#!/usr/bin/bash

#SBATCH -J ecd-run
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=32G
#SBATCH -p batch_ugrad
#SBATCH -t 6-0
#SBATCH -o logs/slurm-%A.out

bash run_ecm.sh 2 6009 --desc bs128.200k.heun --ecd=True --outdir=ecd -heun=True

exit 0
