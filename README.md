# Y-Net Docker  

To learn how to install Docker please visit [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)  

## Nvidia Docker  
If you have never run docker with nvidia-gpu enabled, follow these steps:  
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
```
OK will be displayed normally  
```
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install nvidia-container-toolkit
sudo systemctl restart docker
```

## Running docker container  
GPU Enabled:
`sudo docker run -p 8501:8501 --gpus all --rm -ti --ipc=host jfmontgar/y_net_gr:latest`
CPU Enabled:
`sudo docker run -p 8501:8501 --rm jfmontgar/y_net_gr:latest`

`-p 8501:8501` is the port that the server will be listening on.  
