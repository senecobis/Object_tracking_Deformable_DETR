# Object_tracking_Deformable_DETR

The following is a branch of Trackformer repo : https://arxiv.org/abs/2101.02702
The objective is to compose different models pretrained on RSL excavator dataset and COCO dataset to perform object tracking for different objects than humans

- clone trackformer repo
- delete src and clone this repo inside /coloned_trackformer_repo : eventually change the name of the cloned repo to src
- import detr model trained on excav dataset inside models (folder in the /coloned_trackformer_repo dir), define a folder called "Excav_detr_multi_frame" and put inside the checkpoint and the config (link : https://drive.google.com/file/d/1D81jNJ4W9x1PwHo5NIZwhdNf0UrqbwAQ/view?usp=sharing)
- finally run python src/new_track to reproduce results
