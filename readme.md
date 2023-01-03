# Improving SFMLearner

### System Pipeline:

![Screenshot from 2021-09-17 14-11-46](https://user-images.githubusercontent.com/43991028/133834858-1acab633-6fb8-4881-9744-5abd5b73bf08.png)

TRAINING:

- Paste the TrainingAndValData/ Folder with the prepared kitti data in this folder.
- Enter the folder using cd TrainingAndValData/
- Copy SfMLearnerDatatrain.txt and SfMLearnerDatatrain.txt inside ./TrainingAndValData/SfMLearnerData

To train default version, check if the SfMLearner class from SfMLearner.py is imported in train.py

- cd SfmLearner/
- run `python3 train.py`

To train the modified version, check if the SfMLearner class from SfMLearner_SSIM.py is imported in train.py and perform the

- cd SfmLearner/
- run `python3 train.py`

EVALUATION:
The pose evalution data is already downloaded in ./SfMLearner/kitti_eval

Download the raw odometry data from kitti website and paste it inside ./TrainingAndValData

For testing pose :

- first run test_kitti_pose.py to get predictions, choose the appropriate groundtruth and predictions folder
- navigate to cd kitti_eval
- next run eval_pose to get results.
  Note: in case of running pretrained pose model, change sequence length = 5

For testing depth :

- first run test_kitti_depth.py to get the depth predictions
- navigate to cd kitti_eval
- next run eval_depth to get results.

To visualize depth use the visualize.ipynb notebook in kitti_eval

References for modified architectures:
https://github.com/yzcjtr/GeoNet
