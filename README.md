pip install -r requirements.txt
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch
apt update
apt install vim openssh-server -y
apt install libsparsehash-dev libboost-dev -y
apt install libgl1 libxrender1 libglib2.0-0 -y

# AccMap

## Setup

### Requirements
OS: Ubuntu18.04 <br>
CUDA: 9.0 <br>
cuDNN: 7.6.5 <br>

### Install GPU Driver
```shell
sudo apt update
sudo apt install nvidia-driver-470 -y
sudo reboot
```

### Install CUDA9.0 in Ubuntu18.04
```shell
sudo apt update
sudo apt install g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev -y
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
sudo chmod +x cuda_9.0.176_384.81_linux-run
sudo ./cuda_9.0.176_384.81_linux-run --override
# Then answer some questions
# You are attempting to install on an unsupported configuration. Do you wish to continue? 
# y(es)
# Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 384.81? 
# n(o)
# Install the CUDA 9.0 Toolkit? 
# y(es)
# Do you want to install a symbolic link at /usr/local/cuda?
# y(es)
# Install the CUDA 9.0 Samples?
# y(es)
rm cuda_9.0.176_384.81_linux-run
echo 'export PATH=/usr/local/cuda-9.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
nvcc --version  # check if install CUDA correctly
```





