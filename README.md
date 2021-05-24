# SLAM-OR 

Proposed SLAM algorithm with detection of objects (OR) in the environment. Based on the algorithm proposed by [Luigi Freda] (https://www.luigifreda.com) it has been extended to include the use of YOLO neural networks to recognize objects from the environment and mark them on a 3D map.

## Install 

Clone this repo and its modules by running 

```
$ git clone --recursive https://github.com/PatrykMazurek/SLAM-OR.git
```

#### Install pySLAM in Your Working Python Environment

If you want to run SLAM-OR, run the script

`$ ./install_all.sh` 

or if you use Anaconda enviroment, run 

`$ ./install_all_conda.sh` 

Install these scripts and the most necessary libraries to run the program. then execute scripts to run the program

`main_slam.py`

### KITTI Datasets

SLAM-OR code expects the following structure in the specified KITTI path folder (specified in the section `[KITTI_DATASET]` of the file `config.ini`). : 

1. Download the dataset (grayscale images) from http://www.cvlibs.net/datasets/kitti/eval_odometry.php and prepare the KITTI folder as specified above

2. Select the corresponding calibration settings file (parameter `[KITTI_DATASET][cam_settings]` in the file `config.ini`)

### Object recognition

In this program we used the YOLO detector to recognize objects from the environment. after this [link] (https://upkrakow-my.sharepoint.com/:f:/g/personal/patryk_mazurek_up_krakow_pl/EoI4eFRuanNEnct094OFizkB0vKlazTHIeJ17VEwx7Qjjw?e=IlarrR) there are pre-trained files with weights for versions: YOLO v3, YOLO v3-tiny, YOLO v4 and YOLO v4-tiny

