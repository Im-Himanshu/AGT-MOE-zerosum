# AGT-MOE-zerosum

references

https://arxiv.org/pdf/2310.14188

https://arxiv.org/pdf/1701.06538

```bash

# how to run
srun --account=mpcs52072 --partition=gpu --constrain=a100 --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=06:00:00 --pty bash -i
# activate the environment
source ~/moe_venv/bin/activate
# run the script
python 


```
