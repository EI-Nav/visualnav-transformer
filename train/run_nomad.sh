#!/bin/bash

# PAI平台运行run_nomad.py的脚本

set -e  # 遇到错误立即退出
set -o pipefail  # 管道命令失败时也退出

export WANDB_API_KEY="1e85b8059f54236139f0b5614273212ad19c0ade"

# 设置工作目录（根据实际情况调整）
WORK_DIR="/x2robot_v2/jake/research/visualnav-transformer/train"
cd "${WORK_DIR}" || exit 1

# 激活conda环境（如果使用conda）
# 如果PAI平台已经激活了环境，可以注释掉下面这行
source /x2robot_v2/jake/miniforge3/etc/profile.d/conda.sh
conda activate nomad

# rm -rf /x2robot_v2/jake/research/visualnav-transformer/train/vint_train/data/data_splits/go_stanford/train/dataset_go_stanford.lmdb
# rm -rf /x2robot_v2/jake/research/visualnav-transformer/train/vint_train/data/data_splits/go_stanford/test/dataset_go_stanford.lmdb

# 运行命令
python train.py -c config/x2robot.yaml

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "脚本执行成功！"
else
    echo "脚本执行失败！"
    exit 1
fi

