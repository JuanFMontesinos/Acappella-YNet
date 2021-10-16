# Y-Net

Official implementation of  *A cappella: Audio-visual Singing VoiceSeparation*, British Machine Vision Conference 2021

Project page: [ipcv.github.io/Acappella/](https://ipcv.github.io/Acappella/)

## Running a demo / Y-Net Inference

We provide simple functions to load models with pre-trained weights. Steps:

1. Clone the repo or download y-net>VnBSS>models (models can run as a standalone package)
2. Load a model:

```
from VnBSS import y_net_gr # or from models import y_net_gr 
model = y_net_gr(n=1)
```

Check this [Google Colab]() example

## Training / Using DEV code

The most difficult part is to prepare the dataset as everything is builded upon a very specific format.  
To run training:  
`python run.py -m model_name --workname experiment_name --arxiv_path directory_of_experiments --pretrained_from path_pret_weights`  
You can inspect the argparse at `default.py`>`argparse_default`.  
Possible model names are: `y_net_g`, `y_net_gr`, `y_net_m`,`y_net_r`,`u_net`,`llcp`
