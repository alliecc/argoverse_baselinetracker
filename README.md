[update 04/2020]

The new CVPR2020 argoverse tracking challenge provides detections! 

See our new baseline using the provided, improved detections here: https://github.com/johnwlambert/argoverse_cbgs_kf_tracker

Link to download the new detections: https://s3.amazonaws.com/argoai-argoverse/detections_v1.1b.zip

# Argoverse Baseline Tracker

## 
Baseline tracker code release for the paper **Argoverse: 3D Tracking and Forecasting With Rich Maps**, CVPR 2019.
[[paper]](http://openaccess.thecvf.com/content_CVPR_2019/html/Chang_Argoverse_3D_Tracking_and_Forecasting_With_Rich_Maps_CVPR_2019_paper.html)&nbsp;  [[website]](https://www.argoverse.org/index.html)&nbsp;

<img src="https://github.com/alliecc/argoverse_baselinetracker/blob/master/bev_083.jpg" width="250"> <img src="https://github.com/alliecc/argoverse_baselinetracker/blob/master/bev_085.jpg" width="250"> <img src="https://github.com/alliecc/argoverse_baselinetracker/blob/master/bev_087.jpg" width="250">

## Introduction

This tracker implementation is meant to be a baseline example to demonstrate the use of the map and tracking data in Argoverse dataset. We don't claim its performance to be the best, and we are looking forward to more map-based tracking methods developed using Argoverse in the future.

## Requirements

To run this tracker, please install following requirements:
- [argoverse api](https://github.com/argoai/argoverse-api)
- sklearn
- pyquaternion
- uuid
- mayavi (for 3D visualization)
- MaskRCNN (commit `55796a04ea770029a80cf5933cc5c3f3f6fa59cf`) 
  Follow the [official guide](https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/INSTALL.md) to install
- [pcl 1.8](https://askubuntu.com/questions/916260/how-to-install-point-cloud-library-v1-8-pcl-1-8-0-on-ubuntu-16-04-2-lts-for) and [python-pcl](https://github.com/strawlab/python-pcl) binding
- Download [smallestenclosingcircle.py](https://www.nayuki.io/res/smallest-enclosing-circle/smallestenclosingcircle.py)
- Download [MinimumBoundingBox](https://github.com/BebeSparkelSparkel/MinimumBoundingBox)
- Download Argoverse Tracking dataset from [Argoverse official website](https://www.argoverse.org/data.html).


example:

Command for running this tracker tracker on Argoverse:

```shell
python3 run_tracking.py --path_dataset=/path/to/argoverse-tracking/test --log_id=<log_id>  --path_output=/path/to/output/folder --use_maskrcnn --region_type=roi --use_map_lane --motion_model='const_v' --measurement_model=both --fix_bbox_size --dataset_name=Argoverse
```

You can use `--save_bev_imgs` to print birds-eye-view image as above example or `--show_segmentation` to plot 3D visualization of segmentation result if mayavi is installed. Tracker output format is the same as Argoverse tracking label format.  

## Docker Image

It might be tricky to install all the dependencies, so we provided docker image.
- Docker version 18.09.7, build 2d0083d

To run docker image, first install nvidia-docker and then run following command to build image using the provided DockerFile:
```shell
nvidia-docker build -t baselinetracker docker/
```
After building the image, run following command to start. Mount dataset folder so the data can be accessed in docker environment:
```shell
nvidia-docker run -v /path/to/argoverse-tracking/test:/data  -it baselinetracker:latest
```
And then clone this repo:
```shell
git clone https://github.com/alliecc/argoverse_baselinetracker
cd argoverse_baselinetracker
wget https://www.nayuki.io/res/smallest-enclosing-circle/smallestenclosingcircle.py
```
Here is an example command to start tracker. The tracking output would be stored in /tracking_output.
```shell
python3 run_tracking.py --path_dataset=/data --log_id=0f0d7759-fa6e-3296-b528-6c862d061bdd  --path_output=/tracking_output --use_maskrcnn --region_type=roi --use_map_lane --motion_model='const_v' --measurement_model=both --fix_bbox_size --dataset_name=Argoverse
```






