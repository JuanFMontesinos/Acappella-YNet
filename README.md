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
```
@inproceedings{acappella,
    author    = {Juan F. Montesinos and
                 Venkatesh S. Kadandale and
                 Gloria Haro},
    title     = {A cappella: Audio-visual Singing VoiceSeparation},
    booktitle = {British Machine Vision Conference (BMVC)},
    year      = {2021},

}
```

## Training / Using DEV code
###Training
The most difficult part is to prepare the dataset as everything is builded upon a very specific format.  
To run training:  
`python run.py -m model_name --workname experiment_name --arxiv_path directory_of_experiments --pretrained_from path_pret_weights`  
You can inspect the argparse at `default.py`>`argparse_default`.  
Possible model names are: `y_net_g`, `y_net_gr`, `y_net_m`,`y_net_r`,`u_net`,`llcp`
### Testing
1. Go to `manuscript_scripts` and replace  checkpoint paths by yours  in the testing scripts. 
2. Run: `bash manuscript_scripts/test_gr_r.sh`
3. Replace the paths of `manuscript_scripts/auto_metrics.py` by your experiment_directory path.  
4. Run: `python manuscript_scripts/auto_metrics.py` to visualise results.  

## Code structure  
The code is defined as an importable package `VnBSS` + a configuration package `config` + scripts. 
#### The model
Each effective model is wrapped by a `nn.Module` which takes care of computing the STFT, the mask, returning the waveform
etcetera... This wrapper can be found at `VnBSS`>`models`>`y_net.py`>`YNet`. To get rid of this you can simply inherit the class,
take minimum layers and keep the `core_forward` method, which is the inference step without the miscelanea.  
#### The dataloader  
The dataloader is a multimodal dataloader. Basically, it inpects a directory and generates a tree. It assumes each children 
folder is a different modality (in this case there will be audio and video). It pairs the files for all the modalities 
and offers them to the user automatically. A path-exclusion system allows ignoring certain folders permitting to create 
different subsets easily. 


## FAQs  
1. *How to change the optimizer's hyperparameters?*  
Go to `config`>`optimizer.json`  
2. *How to change clip duration, video framerate, STFT parameters or audio samplerate?*  
Go to `config`>`__init__.py`  
3. *How to change the batch size or the amount of epochs?*  
Go to `config`>`hyptrs.json`