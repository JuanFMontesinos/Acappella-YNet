# Understanding the Dataset structure and Dataloader
The dataloader is a multimodal dataloader. Given a directory, the children directories
are assumed to be each possible modality. For this work we will find:
```
--Acappella
  |
  |---audio
  |   |
  |   |--English
  |   |--Hindi
  |   |--Others
  |   |--Spanish 
  |
  |---frames  
  |   |
  |   |--English
  |   |--Hindi
  |   |--Others
  |   |--Spanish 
  |
  |---landmarks  
  |   |
  |   |--English
  |   |--Hindi
  |   |--Others
  |   |--Spanish 
  |
  |---llcp_embed  
  |   |
  |   |--English
  |   |--Hindi
  |   |--Others
  |   |--Spanish 
  |
  
```

Each language folder will contain male and female categories and so on.  audio, frames, landmarks and llcp_embed
are the modalities.  

* The dataset class  ( `BaseDataHandler` at `dataset_helpers`>`helpers.py`) is a dynamic dataset.
It will match the samples for each modality and will offer to the user automatically.  
* It is **reproducible**: each multimodal set is determined by an ID and the arguments to read the file.
Thus, it allows to keep track of each sample used and to recover network's predictions if needed.
  
Each modality has an associated reader (for images, numpy arrays, mat files...)
