#! /bin/sh

#SBATCH --job-name=run_vicuna_eval
#SBATCH --account=gpu-students
#SBATCH --output=run_vicuna_eval.out # redirect stdout
#SBATCH --error=run_vicuna_eval.err # redirect stderr
#SBATCH --partition=studentkillable 
#SBATCH --time=1440 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total

python eval_on_dataset.py -m "lmsys/vicuna-7b-v1.3" -d "updated_examples.csv"

