

OS="ubuntu2004"
ARCH=x86_64
cudnn_version="8.2.4.*"
cuda_version="cuda11.4"

# # echo ${OS}
# sudo dpkg -i cuda-repo-ubuntu2004_x86_64.deb

wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin 
sudo mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt-get update

# # echo "deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/ /"

sudo apt-get install -y cuda-11-4

sudo apt-get install -y libcudnn8=${cudnn_version}-1+${cuda_version}
sudo apt-get install -y libcudnn8-dev=${cudnn_version}-1+${cuda_version}