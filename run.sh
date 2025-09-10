#!/bin/bash
#SBATCH --job-name=titan_ds_v3_train     # Job name
#SBATCH --nodes=16                        # Number of nodes
#SBATCH --ntasks=16                        # Number of nodes
#SBATCH --gpus-per-task=8              # Number of tasks per node (e.g., 8 GPUs per node)
#SBATCH --time=48:00:00                  # Walltime (hh:mm:ss)
#SBATCH --partition=p5-cb-queue          # Partition name
#SBATCH --output=slurm-%j.out          	 # Stdout file
#SBATCH --chdir=/home/ubuntu/torchtitan

#SBATCH --cpus-per-task=96

# Master node and port
nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

# Optional: reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True"

# Each GPU gets its local rank
export LOG_RANK=$SLURM_LOCALID
export NGPU=8

export FI_PROVIDER="efa"
export NVSHMEM_LIBFABRIC_PROVIDER=efa
export NVSHMEM_REMOTE_TRANSPORT=libfabric

export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_BUFFSIZE=2097152
#export TORCH_DIST_INIT_BARRIER=1
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0


CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml"}

dcgmi profile --pause
# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.

srun torchrun --nnodes 16 --nproc_per_node 8 --rdzv_id 101 --rdzv_backend c10d --rdzv_endpoint "$head_node_ip:29500" \
    ./torchtitan/train.py --job.config_file ${CONFIG_FILE} "$@" \
    --parallelism.pipeline_parallel_degree 1 \
    --parallelism.tensor_parallel_degree 8 \
    --parallelism.expert_parallel_degree 8 \
    --training.local_batch_size 16 \
    --profiling.no-enable-profiling
    # --model.converters="float8" \
    # --compile.components model loss \
    # --float8.recipe_name="rowwise"
    # --float8.enable_fsdp_float8_all_gather \

    # --parallelism.expert_tensor_parallel_degree 8 \

dcgmi profile --resume
