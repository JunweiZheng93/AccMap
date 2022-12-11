# AccMap

## Demo Video
[![MapAugmentation Demo Video](https://res.cloudinary.com/marcomontalbano/image/upload/v1670718422/video_to_markdown/images/youtube--R3bbKG-te7k-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=R3bbKG-te7k "MapAugmentation Demo Video")

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

### Install cuDNN7.6.5 in Ubuntu18.04
Download cuDNN7.6.5 from [here](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/9.0_20191031/cudnn-9.0-linux-x64-v7.6.5.32.tgz), and unzip it to current working directory. You may need to sign up in order to download cuDNN.
```shell
tar -xzvf cudnn-9.0-linux-x64-v7.6.5.32.tgz  # unzip the file
sudo cp -P cuda/include/cudnn.h /usr/local/cuda-9.0/include  # copy files to /usr/local/cuda-9.0/include
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64/  # copy files to /usr/local/cuda-9.0/lib64/
sudo chmod a+r /usr/local/cuda-9.0/lib64/libcudnn*
rm cudnn-9.0-linux-x64-v7.6.5.32.tgz  # delete the zip file
rm -rf cuda/  # delete the unzip file
```

### Install conda
```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh
rm Miniconda3-py39_4.12.0-Linux-x86_64.sh
source ~/.bashrc
conda create -n AccMap python=3.7 -y
echo 'conda activate AccMap' >> ~/.bashrc
source ~/.bashrc
```

### Install all necessary dependencies
```shell
git clone https://github.com/JunweiZheng93/AccMap.git
cd AccMap
pip install -r requirements.txt
conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=9.0 -c pytorch -y
```

Compile and install `spconv`:
```shell
sudo apt install libboost-dev -y
sudo apt install gcc-5 g++-5 -y
sudo ln -s /usr/bin/gcc-5 /usr/local/cuda/bin/gcc  # create symlink because compiling spconv require gcc-5
sudo ln -s /usr/bin/g++-5 /usr/local/cuda/bin/g++  # create symlink because compiling spconv require g++-5
cd lib/spconv
python setup.py bdist_wheel
cd dist
pip install *
```

Compile and install `pg_op`:
```shell
sudo apt install libsparsehash-dev -y
cd lib/pointgroup_ops
python setup.py develop
```



