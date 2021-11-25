# Understanding the Dataset structure and Dataloader
The dataloader is a multimodal dataloader. Given a directory, the children directories
are assumed to be each possible modality. For this work we will find:
```
--Acappella  
  |---train
  |   |---audio
  |   |   |--English
  |   |   |  |
  |   |   |  |--Male
  |   |   |  |  <sample_id>.wav
  |   |   |  |  <sample_id>.wav
  |   |   |  |  ...  
  |   |   |  |--Female
  |   |   |  |  ...  
  |   |   |--Spanish
  |   |   |  ...
  |   |   |--Hindi
  |   |   |  ...
  |   |   |--Others
  |   |   |  ...
  |   |---frames
  ...
  |   |---landmarks
  ...
  |---train
  |...
  |---test_unseen
  |...
  |---test_seen
  |...
  |---val_seen
  
```

Each language folder will contain male and female categories and so on.  audio, frames, landmarks and llcp_embed
are the modalities.  

* The dataset class  ( `BaseDataHandler` at `dataset_helpers`>`helpers.py`) is a dynamic dataset.
It will match the samples for each modality and will offer to the user automatically.  
* It is **reproducible**: each multimodal set is determined by an ID and the arguments to read the file.
Thus, it allows to keep track of each sample used and to recover network's predictions if needed.
  
Each modality has an associated reader (for images, numpy arrays, mat files...)

In case of audioset it follows the same distribution:

--Audioset
  |--eval
  |   |--audio
  |   |  |---<audio_id>.wav
  |   |  |---<audio_id>.wav
  |   |   |---...
  |   |  |--Beatboxing
  |   |   |---...
  |   |  |--Choir
  |   |   |---...
  |   |  |--Backgroundmusic
  |   |  |--Drum
  |   |  |--Rapping
  |          ...
  |   
  |
  |--unbalanced_train
  |   |--audio
  |   |  |
  |   |  |--Beatboxing
  |   |  |--Choir
  |   |  |--Backgroundmusic
  |   |  |--Drum
  |   |  |--Rapping
  |          ...
  |--test
  |   |--audio
  ...