conda create -n nomad python=3.10
conda activate nomad

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 -i https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com

pip install wandb warmup_scheduler diffusers efficientnet_pytorch einops vit_pytorch lmdb prettytable matplotlib opencv-python fastapi uvicorn gsplat plyfile imageio imageio[ffmpeg]