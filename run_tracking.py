"""main file for running argoverse baseline tracker"""
import numpy as np
import argparse
import time
import os
import cv2
import glob

from argoverse.data_loading.synchronization_database import SynchronizationDB
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.transform import quat2rotmat
from argoverse.utils.se3 import SE3
from argoverse.utils.ply_loader import load_ply
from tracker import Tracker, Detector
from tracker_tools import (
    leave_only_roi_region,
    leave_only_driveable_region,
    show_pc_segments,
)

import sys
from pykitti import tracking as kitti_tracking
sys.path.append('pykitti')

def run_tracking(
    dataset_name,
    dataset_dir,
    log_id,
    path_output,
    ground_level,
    ground_removal_th,
    use_map_lane,
    use_maskrcnn,
    min_point_num,
    motion_model,
    fix_bbox_size,
    dbscan_eps,
    ground_removal_method,
    log_images,
    maskrcnn_model,
    show_segmentation,
    region_type,
    save_bev_imgs
):
    """Main function for runnning baseline tracker.

    Args:
        datasetname: supports Argoverse and KITTI
        dataset_dir: root path of input dataset
        log_id: id of the input
        path_output: root path of tracking output 
        ground_level: ground height for threshold-based ground removal method
        ground_removal_th: ground segmentation threshold for plane-fitting ground removal
        use_map_lane: use map center line direction to help tracking
        use_maskrcnn: use MaskRCNN to filter segmentation result
        min_point_num: minimun number of points for tracked point cloud segment 
        motion_model: "static", "const_v", "measure_only"
        fix_bbox_size: boolean value for using fixed bbox size or use tight bbox 
        dbscan_eps: eps parameter for DBSCAN segmentation
        ground_removal_method: "map", "plane_fitting", "threshold"
        log_images: save detection result as image output
        maskrcnn_model: path to the MaskRCNN trained model
        show_segmentation: boolean value for showing semengtation result in 3D view or not
        region_type: prefiltering input lidar point cloud using map "driveable", "roi", "all"
        save_bev_imgs: save Bird's-eye view image

    Returns:
        No returns. Results would be saved to path_output
    """

    
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    
    #Initialize dataset 
    if dataset_name == 'KITTI':
        if ground_removal_method == 'map' or use_map_lane == True or region_type != "all":
            print('KITTI has no map data! Turn off map-based functions.')
            raise NotImplementedError

        kitti_data = kitti_tracking(dataset_dir, log_id)
        path_input = os.path.join(dataset_dir)
        city_name = None
        calib_data = kitti_data.get_calib()
        T_w_to_imu_list = kitti_data.get_oxt_list()
        num_frames = kitti_data.__len__()

    elif dataset_name == 'Argoverse':
        path_input = os.path.join(dataset_dir, log_id)
        city_info_fpath = f"{dataset_dir}/{log_id}/city_info.json"
        city_info = read_json_file(city_info_fpath)
        city_name = city_info["city_name"]
        calib_fpath = f"{path_input}/vehicle_calibration_info.json"
        calib_data = read_json_file(calib_fpath)

        sdb = SynchronizationDB(dataset_dir, collect_single_log_id=log_id)
    
        path_lidars = glob.glob(os.path.join(path_input, "lidar/*.ply"))
        path_lidars.sort()
        num_frames = len(path_lidars)
    else:
         raise NotImplementedError

    #Initialize detector and tracker 
    detector = Detector(
        region_type,
        ground_level,
        ground_removal_th,
        use_maskrcnn,
        city_name,
        min_point_num,
        dbscan_eps,
        calib_data,
        path_output,
        ground_removal_method,
        dataset_name,
        maskrcnn_model,
    )
    tracker = Tracker(
        motion_model,
        measurement_model,
        use_map_lane,
        fix_bbox_size,
        city_name,
        path_output,
    )


    for ind_frame in range(num_frames):

        list_img = []

        if dataset_name == 'KITTI':
            pc_raw_roi = kitti_data.get_velo(ind_frame)[:, 0:3]
            T_w_to_imu = T_w_to_imu_list[ind_frame]
            T_imu_to_velo = calib_data['Tr_imu_velo']
            T_w_to_velo = np.matmul(T_w_to_imu, T_imu_to_velo)

            ego_R =  T_w_to_velo[0:3,0:3].transpose()
            ego_t = - np.matmul(ego_R, T_w_to_velo[0:3,3][:,np.newaxis])[:,0]
            egovehicle_to_city_se3 = SE3(rotation=ego_R, translation=ego_t)

            lidar_timestamp =  ind_frame #fake lidar timestamp for KITTI

            if use_maskrcnn:
                list_img.append( cv2.cvtColor(np.array(kitti_data.get_cam2(ind_frame)), cv2.COLOR_RGB2BGR))

        elif dataset_name == 'Argoverse':
    
            lidar_timestamp = int(
                path_lidars[ind_frame].split("/")[-1].split("_")[-1].split(".")[-2]
            )
            print("Processing frame %s, time stamp = %d" % (ind_frame, lidar_timestamp))
            pc_raw0 = load_ply(
                path_lidars[ind_frame]
            )  
    
            pose_path = (
                f"{dataset_dir}/{log_id}/poses/city_SE3_egovehicle_{lidar_timestamp}.json"
            )
    
            if not os.path.exists(pose_path):
                print("Missing ", pose_path)
                print("Skip this frame....")
                continue

            pose_data = read_json_file(pose_path)
            rotation = np.array(pose_data["rotation"])
            translation = np.array(pose_data["translation"])
            ego_R = quat2rotmat(rotation)
            ego_t = translation
            egovehicle_to_city_se3 = SE3(rotation=ego_R, translation=ego_t)
    
    
            if region_type == "driveable":
                pc_raw_roi = leave_only_driveable_region(
                    pc_raw0,
                    egovehicle_to_city_se3,
                    ground_removal_method=ground_removal_method,
                    city_name=city_name,
                )
    
            elif region_type == "roi":
                pc_raw_roi = leave_only_roi_region(
                    pc_raw0,
                    egovehicle_to_city_se3,
                    ground_removal_method=ground_removal_method,
                    city_name=city_name,
                )
    
            elif region_type == "all":
                pc_raw_roi = pc_raw0.copy()
            else:
                print("Error! Region type not implemented")
    
           
            if use_maskrcnn:
                for ind_cam in range(len(detector._RING_CAMERA_LIST)):
                    camera_name = detector._RING_CAMERA_LIST[ind_cam]
                    cam_timestamp = sdb.get_closest_cam_channel_timestamp(
                        lidar_timestamp, camera_name, log_id
                    )
                    if cam_timestamp == None:
                        print("Cannot find %s image from timestamp!!" % camera_name)
                        return
    
                    path_img = os.path.join(
                        path_input, camera_name, "%s_%d.jpg" % (camera_name, cam_timestamp)
                    )
    
                    if not os.path.exists(path_img):
                        print("Missing image!", path_img, "ignore mask rcnn...")
                        
                        return
    
                    list_img.append(cv2.imread(path_img))

        #do point cloud segmentation to get object candidates
        pc_segments = detector.get_candidate_segments(
            pc_raw_roi, list_img, ind_frame, log_images
        )
        print("Detected %d objects" % len(pc_segments))

        #function for visualizing detection result
        if show_segmentation:
            if dataset_name == 'Argoverse': #use different colors for raw pc and roi pc
                show_pc_segments(pc_raw0, pc_raw_roi, pc_segments)
            else:
                show_pc_segments(pc_raw_roi, pc_raw_roi, pc_segments)


        # transform segmentation results to global coordinate
        for ii in range(len(pc_segments)):
            pc_segments[ii] = egovehicle_to_city_se3.transform_point_cloud(
                pc_segments[ii]
            )  

        if len(tracker.tracks) == 0:
            tracker.initialize_tracks(pc_segments, ego_R, ego_t)
        else:
            tracker.find_matches(pc_segments)
            tracker.update(pc_segments, ego_R, ego_t)

        tracker.save_result_label_format(lidar_timestamp, egovehicle_to_city_se3, dataset_name)
        if save_bev_imgs:

            tracker.save_bev_img(dataset_name, log_id, lidar_timestamp, pc_raw_roi, egovehicle_to_city_se3)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ground_removal_th",
        type=float,
        help="threshold for ground removal height",
        default=0.3,
    )
    parser.add_argument("--ground_level", type=float, help="ground height", default=0.3)
    parser.add_argument(
        "--dbscan_eps", type=float, help="eps parameter for dbscan", default=1.0
    )
    parser.add_argument(
        "--path_dataset", type=str, help="root path for input dataset", required=True
    )
    parser.add_argument(
        "--log_id", type=str, help="root path for input dataset", required=True
    )

    parser.add_argument(
        "--path_output", type=str, help="root path for output files", required=True
    )
    parser.add_argument(
        "--min_point_num",
        type=float,
        help="minimum number of points in a segment",
        default=30,
    )
    parser.add_argument(
        "--show_segmentation",
        help="show db scan result",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--motion_model",
        type=str,
        help="motion model in KF",
        default="static",
        choices=["static", "const_v", "measure_only"],
    )
    parser.add_argument(
        "--measurement_model",
        type=str,
        help="motion model in KF",
        default="detect",
        choices=["icp", "detect", "both"],
    )
    parser.add_argument(
        "--ground_removal_method",
        type=str,
        help="method for ground removal",
        default="map",
        choices=["map", "plane_fitting", "threshold"],
    )
    parser.add_argument(
        "--region_type",
        type=str,
        help="driveable or ROI or all",
        default="driveable",
        choices=["driveable", "roi", "all"],
    )
    parser.add_argument(
        "--use_map_lane",
        help="use map lane direction",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--fix_bbox_size",
        help="use same bbox size for every car",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--use_maskrcnn",
        help="use Mask R-CNN to filter detection",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--log_images",
        help="save detection result as images",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--maskrcnn_model",
        type=str,
        help="path to mask rcnn model weights",
        default="maskrcnn_model/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        help="name of input dataset",
        default="Argoverse",
        choices=["Argoverse", "KITTI"]
    )

    parser.add_argument(
        "--save_bev_imgs",
        help="save birds eye view image",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()
    print(args)

    dataset_name = args.dataset_name
    ground_level = args.ground_level
    dbscan_eps = args.dbscan_eps
    path_dataset = args.path_dataset
    log_id = args.log_id
    min_point_num = args.min_point_num
    ground_removal_th = args.ground_removal_th
    motion_model = args.motion_model
    use_map_lane = args.use_map_lane
    fix_bbox_size = args.fix_bbox_size
    use_maskrcnn = args.use_maskrcnn
    measurement_model = args.measurement_model
    ground_removal_method = args.ground_removal_method
    log_images = args.log_images
    maskrcnn_model = args.maskrcnn_model
    show_segmentation = args.show_segmentation
    region_type = args.region_type
    save_bev_imgs = args.save_bev_imgs

    path_output = os.path.join(
        args.path_output,
        log_id
        + ("_lane_%d_fixbbox_%d_rcnn_%d_%s_%s_%s_%s")
        % (
            use_map_lane,
            fix_bbox_size,
            use_maskrcnn,
            ground_removal_method,
            motion_model,
            measurement_model,
            region_type,
        ),
    )

    run_tracking(
        dataset_name,
        path_dataset,
        log_id,
        path_output,
        ground_level,
        ground_removal_th,
        use_map_lane,
        use_maskrcnn,
        min_point_num,
        motion_model,
        fix_bbox_size,
        dbscan_eps,
        ground_removal_method,
        log_images,
        maskrcnn_model,
        show_segmentation,
        region_type,
        save_bev_imgs
    )

