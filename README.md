# Y-Net

Official implementation of  *A cappella: Audio-visual Singing VoiceSeparation*, British Machine Vision Conference 2021

Project page: [ipcv.github.io/Acappella/](https://ipcv.github.io/Acappella/)  
Paper: [Arxiv](https://arxiv.org/abs/2104.09946), [Supplementary Material](https://raw.githubusercontent.com/IPCV/Acappella/master/supplementary_material.pdf),
BMVC (not available yet)  


## Running a demo / Y-Net Inference

We provide simple functions to load models with pre-trained weights. Steps:

1. Clone the repo or download y-net>VnBSS>models (models can run as a standalone package)
2. Load a model:

```
from VnBSS import y_net_gr # or from models import y_net_gr 
model = y_net_gr()
```
Examples can be found at `y_net`>`examples`. Also you can have a look at `tcol.py` or `example.py`, files which 
computes the demos shown in the website.  
Check a demo fully working:  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1jFDy9vkuXDqyS63y0SCHkNTb7p494fSp?usp=sharing)

## Citation
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
.  
.  
.  
.  
.  
.  

# Training / Using DEV code
### Training
The most difficult part is to prepare the dataset as everything is builded upon a very specific format.  
Download the code and set your dataset paths at `config>dataset_paths.json`  
To run training:  
`python run.py -m model_name --workname experiment_name --arxiv_path directory_of_experiments --pretrained_from path_pret_weights`  
You can inspect the argparse at `default.py`>`argparse_default`.  
Possible model names are: `y_net_g`, `y_net_gr`, `y_net_m`,`y_net_r`,`u_net`,`llcp`
### Testing
1. Go to `manuscript_scripts` and replace  checkpoint paths by yours  in the testing scripts. 
2. Run: `bash manuscript_scripts/test_gr_r.sh`
3. Replace the paths of `manuscript_scripts/auto_metrics.py` by your experiment_directory path.  
4. Run: `python manuscript_scripts/auto_metrics.py` to visualise results.  

### It's a complicated framework. HELP!
The best option to run the framework is to debug! Having a runable code helps to see input shapes, dataflow and
to run line by line. Download [The circle of life](https://ipcv.github.io/Acappella/dataset/) demo with the files
already processed. It will act like a dataset of 6 samples. You can download it from
[Google Drive](https://drive.google.com/file/d/1An3kalwUpyPWpeH_urJchWsWaffVj3_J/view?usp=sharing) 1.1 Gb.
1. Unzip the file  
2. run `python run.py -m y_net_gr` (for example) TODO :D   

Everything has been configured to run by default this way.


#### The model
Each effective model is wrapped by a `nn.Module` which takes care of computing the STFT, the mask, returning the waveform
etcetera... This wrapper can be found at `VnBSS`>`models`>`y_net.py`>`YNet`. To get rid of this you can simply inherit the class,
take minimum layers and keep the `core_forward` method, which is the inference step without the miscelanea.  

## Downloading the datasets  
To download the Acappella Dataset run the script at `preproc`>`preprocess.py`  
To download the demos used in the website run `preproc`>`demo_preprocessor.py`  
Audioset can be downloaded via webapp, `streamlit run audioset.py`  
## Computing the demos  
Demos shown in the website can be computed:
* **The circle of life** demo is obtained by running `tcol.py`. First turn the flag `COMPUTE=True`. To visualize
the results turn the flag `COMPUTE=False` and run a `streamlit run tcol.py`.  
  

## FAQs  
1. *How to change the optimizer's hyperparameters?*  
Go to `config`>`optimizer.json`  
2. *How to change clip duration, video framerate, STFT parameters or audio samplerate?*  
Go to `config`>`__init__.py`  
3. *How to change the batch size or the amount of epochs?*  
Go to `config`>`hyptrs.json`  
4. *How to dump predictions from the training and test set*  
Go to `default.py`. Modify `DUMP_FILES` (can be controlled at a subset level). `force` argument 
   skips the iteration-wise conditions and dumps for every single network prediction.  
5. *Is tensorboard enabled?*  
Yes, you will find tensorboard records at `your_experiment_directory/used_workname/tensorboard`  
6. *Can I resume an experiment?*  
Yes, if you set exactly the same experiment folder and workname, the system will detect it and will resume from there.  
7. *I'm trying to resume but found `AssertionError`*
If there is an exception before running the model
8. *How to change the amount of layers of U-Net*  
U-net is  build dynamically given a list of layers per block as shown in `models`>`__init__.py` from outer to inner blocks.  
9. *How to modify the default network values?*  
The json file `config`>`net_cfg.json` overwrites any default configuration from the model. 