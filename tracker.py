import os
import sys
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import random
import torch
import json
import random
import glob
import pickle
import argparse
import time
import cv2
import copy
import pickle

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from argoverse.utils.se3 import SE3
from MinimumBoundingBox import MinimumBoundingBox
from scipy.optimize import linear_sum_assignment
from tracker_tools import (
    pc_remove_ground,
    point_cloud_to_homogeneous,
    project_lidar_to_img,
    pc_segmentation_dbscan_multilevel,
    get_color,
    is_car,
    sort_pc_segments_by_area,
    check_pc_overlap,
    merge_pcs,
    generate_ID,
    get_polygon_center,
    initialize_bbox,
    check_pc_bbox_overlap,
    get_icp_measurement,
    get_z_icp,
    get_z_detect,
    do_tracking,
    compute_bbox,
    check_orientation,
    rotate_orientation,
    build_bbox,
    project_lidar_to_img_kitti
)


#parameters for visualization
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1.5
fontColor = (0, 0, 255)
lineType = 2
color_dict = {}

class Tracker:
    """
    Main class for baseline tracker
    """
    motion_model = "static"
    measurement_model = "detect"
    use_map_lane = True  # use map lane direction
    dataset_type = 0
    fix_bbox_size = True  # use same bbox size for every car
    tracks = []
    pc_matched_id = {}
    ind_current_frame = 0
    city_name = 5
    count_fail_th = 5
    path_output = ""

    def __init__(
        self,
        motion_model="static",
        measurement_model="detect",
        use_map_lane=True,
        fix_bbox_size=True,
        city_name="MIA",
        path_output="",
    ):
        self.motion_model = motion_model
        self.measurement_model = measurement_model
        self.use_map_lane = use_map_lane
        self.fix_bbox_size = fix_bbox_size
        self.ind_current_frame = 0
        self.city_name = city_name
        self.path_output = path_output

    def initialize_tracks(self, pc_segments, city_R_egovehicle, city_t_egovehicle):
        """
        Use point cloud segments to initialize tracks
        """
        self.tracks.append({})

        for ii in range(len(pc_segments)):
            new_id = generate_ID()
            self.add_new_track(
                pc_segments[ii], new_id, city_R_egovehicle, city_t_egovehicle
            )


    def find_matches(self, pc_candidates):
        """
        Do object association.
        1. Find corresponding objects in previous frame
        2. returns an array with same length as new detections, storing correspounding index in tracks[i-1].keys()
        3. If no matched is found, the correspoinding index is -1
        """
        tracks = self.tracks
        ind_current_frame = self.ind_current_frame
        tracked_id = list(tracks[ind_current_frame].keys())
        track_matched = np.zeros(len(tracked_id))
        pc_matched_id = -np.ones(len(pc_candidates))
        match_label_ind = -np.ones(len(tracked_id))

        # loop through all pairs of candidates and existing tracks to find matches
        for ii in range(len(pc_candidates)):
            ind_match_min = -1
            dist_min = np.inf

            for iii in range(len(tracked_id)):

                if (
                    track_matched[iii] == 1
                    or tracks[ind_current_frame][tracked_id[iii]]["count_fail"]
                    >= self.count_fail_th
                ):
                    continue

                center_seg = get_polygon_center(pc_candidates[ii])
                is_close = False

                bbox_min = MinimumBoundingBox.MinimumBoundingBox(pc_candidates[ii][:, 0:2])
                l01 = bbox_min.length_parallel
                l02 = bbox_min.length_orthogonal

                diff = (tracks[ind_current_frame][tracked_id[iii]]["x"][0:2] - center_seg[0:2])
                dist_center = np.linalg.norm(diff)
                theta = tracks[ind_current_frame][tracked_id[iii]]["x"][3]
                vec_parallel = np.array([np.cos(theta), np.sin(theta)] )  
                if np.dot(diff, vec_parallel) < 0:
                    vec_parallel = -vec_parallel
                diff_parallel = np.dot(diff / np.linalg.norm(diff), vec_parallel) * diff
                diff_vertical = diff - diff_parallel

                #considering car motion model, use different distance threshold for vertical and parallel directions
                if (np.linalg.norm(diff_parallel) < 1.0
                    + np.linalg.norm(tracks[ind_current_frame][tracked_id[iii]]["vx"])
                    and np.linalg.norm(diff_vertical) < 1):
                    is_close = True

                overlap = check_pc_bbox_overlap(
                    pc_candidates[ii],
                    tracks[ind_current_frame][tracked_id[iii]]["bbox"],
                )

                
                if is_close or overlap:
                    if dist_center < dist_min:
                        ind_match_min = iii
                        dist_min = dist_center

            if ind_match_min != -1:
                track_matched[ind_match_min] = 1
                pc_matched_id[ii] = ind_match_min

            else:
                print("segment ", ii, ": match not found!!!!")

        pc_matched_id = pc_matched_id.astype("int")

        # select the segments that matched to new detection result
        pc_segments_selected = np.array(pc_candidates)[pc_matched_id != -1]
        pc_matched_id_selected = pc_matched_id[pc_matched_id != -1]

        num_matched = len(pc_matched_id_selected)
        dist_matrix = np.zeros((num_matched, num_matched))

        # find global optimum assignment
        for ii in range(num_matched):
            for jj in range(num_matched):
                pc_target = pc_segments_selected[ii]
                center_target = get_polygon_center(pc_target)

                center_track_keep = tracks[ind_current_frame][
                    tracked_id[pc_matched_id_selected[jj]]
                ]["x"]
                dist_matrix[ii, jj] = np.sum(
                    (center_target[0:2] - center_track_keep[0:2]) ** 2
                )

        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        pc_matched_id_selected[row_ind] = pc_matched_id_selected[col_ind]
        pc_matched_id[pc_matched_id != -1] = pc_matched_id_selected
        self.pc_matched_id = pc_matched_id


    def update(self, pc_candidates, city_R_egovehicle, city_t_egovehicle):
        """
        update tracked object state given the association result
        """
        if self.ind_current_frame > 5:
            self.tracks[self.ind_current_frame - 5] = 0

        # frame index of current frame
        self.ind_current_frame += 1
        ind_current_frame = self.ind_current_frame
        pc_matched_id = self.pc_matched_id
        tracks = self.tracks
        tracks.append({})
        tracked_id = list(tracks[ind_current_frame - 1].keys())

        print("update tracker: frame %05d" % ind_current_frame)
        for ii in range(len(pc_matched_id)):

            # No match found!! initialize new track
            if pc_matched_id[ii] == -1:
                new_id = generate_ID()
                self.add_new_track(
                    pc_candidates[ii], new_id, city_R_egovehicle, city_t_egovehicle
                )

            # match found, update existing track!
            else:
                matched_ind = pc_matched_id[ii]
                print("Update object ", tracked_id[matched_ind])
                tracks[ind_current_frame][tracked_id[matched_ind]] = {}

                tracks[ind_current_frame][tracked_id[matched_ind]]["tracked"] = True
                tracks[ind_current_frame][tracked_id[matched_ind]]["num_points"] = len(pc_candidates[ii])
                tracks[ind_current_frame][tracked_id[matched_ind]]["count"] = (
                    tracks[ind_current_frame - 1][tracked_id[matched_ind]]["count"] + 1
                )
                tracks[ind_current_frame][tracked_id[matched_ind]][
                    "pc_world"
                ] = pc_candidates[ii]
                tracks[ind_current_frame][tracked_id[matched_ind]]["count_fail"] = 0

                pc_seg = pc_candidates[ii]
                center_seg = get_polygon_center(pc_seg)

                # use icp to propogate pose estimation as one measurement model
                transf, icp_fitness, pc_accu = get_icp_measurement(
                    tracks[ind_current_frame - 1][tracked_id[matched_ind]]["pc_world"],
                    tracks[ind_current_frame][tracked_id[matched_ind]]["pc_world"],
                )

                # compute icp pose estimation from incrementing previous frame pose
                z_icp = get_z_icp(
                    tracks[ind_current_frame - 1][tracked_id[matched_ind]]["x"], transf
                )

                tracks[ind_current_frame][tracked_id[matched_ind]]["center_pc"] = get_polygon_center(
                    tracks[ind_current_frame][tracked_id[matched_ind]]["pc_world"]
                )

                #use detection as another measurement model
                z_detect, theta_conf_detect = get_z_detect(
                    tracks[ind_current_frame][tracked_id[matched_ind]]["pc_world"],
                    tracks[ind_current_frame - 1][tracked_id[matched_ind]]["vx"],
                    self.use_map_lane,
                    city_R_egovehicle,
                    city_t_egovehicle,
                    self.city_name,
                )

                if tracks[ind_current_frame][tracked_id[matched_ind]]["count"]<2:
                    tracks[ind_current_frame][tracked_id[matched_ind]]["x"], tracks[
                    ind_current_frame][tracked_id[matched_ind]]["Sigma"] = do_tracking(
                    tracks[ind_current_frame - 1][tracked_id[matched_ind]]["x"],
                    tracks[ind_current_frame - 1][tracked_id[matched_ind]]["vx"],
                    tracks[ind_current_frame - 1][tracked_id[matched_ind]]["Sigma"],
                    z_detect,
                    theta_conf_detect,
                    z_icp,
                    icp_fitness,
                    tracks[ind_current_frame][tracked_id[matched_ind]]["pc_world"],
                    'measure_only',
                    self.measurement_model,
                )

                else:
                    tracks[ind_current_frame][tracked_id[matched_ind]]["x"], tracks[
                    ind_current_frame][tracked_id[matched_ind]]["Sigma"] = do_tracking(
                    tracks[ind_current_frame - 1][tracked_id[matched_ind]]["x"],
                    tracks[ind_current_frame - 1][tracked_id[matched_ind]]["vx"],
                    tracks[ind_current_frame - 1][tracked_id[matched_ind]]["Sigma"],
                    z_detect,
                    theta_conf_detect,
                    z_icp,
                    icp_fitness,
                    tracks[ind_current_frame][tracked_id[matched_ind]]["pc_world"],
                    self.motion_model,
                    self.measurement_model,
                )

                tracks[ind_current_frame][tracked_id[matched_ind]][
                    "bbox"
                ], angle = compute_bbox(
                    tracks[ind_current_frame][tracked_id[matched_ind]]["pc_world"],
                    city_R_egovehicle,
                    city_t_egovehicle,
                    self.city_name,
                    self.use_map_lane,
                    x_initial=tracks[ind_current_frame][tracked_id[matched_ind]]["x"],
                    fix_size=self.fix_bbox_size,
                )

                # estimate velocity from previous frames
                if tracks[ind_current_frame][tracked_id[matched_ind]]["count"] < 2:
                    tracks[ind_current_frame][tracked_id[matched_ind]]["vx"] = tracks[
                        ind_current_frame - 1
                    ][tracked_id[matched_ind]]["vx"]

                else:
                    # assume no sudden change in velocity
                    update_ratio = 0.5
                    tracks[ind_current_frame][tracked_id[matched_ind]]["vx"] = (
                        update_ratio
                        * (
                            (
                                tracks[ind_current_frame][tracked_id[matched_ind]]["x"]
                                - tracks[ind_current_frame - 1][tracked_id[matched_ind]]["x"]
                            )
                        )
                        + (1 - update_ratio)
                        * tracks[ind_current_frame - 1][tracked_id[matched_ind]]["vx"]
                    )

                    tracks[ind_current_frame][tracked_id[matched_ind]]["x"][3] = check_orientation(
                        tracks[ind_current_frame][tracked_id[matched_ind]]["vx"],
                        tracks[ind_current_frame][tracked_id[matched_ind]]["x"][3],
                    )

        # update those tracked but not associated objects using motion model
        for ii in range(len(tracked_id)):

            if ii in pc_matched_id:
                continue

            print("Missing object ", tracked_id[ii])

            #Remove objects that detected for a while
            if (tracks[ind_current_frame - 1][tracked_id[ii]]["count_fail"]
                >= self.count_fail_th):
                print("Removing object ", tracked_id[ii])  
                continue

            tracks[ind_current_frame][tracked_id[ii]] = {}
            tracks[ind_current_frame][tracked_id[ii]]["center_pc"] = tracks[
                ind_current_frame - 1
            ][tracked_id[ii]]["center_pc"]

            tracks[ind_current_frame][tracked_id[ii]]["vx"] = tracks[
                ind_current_frame - 1
            ][tracked_id[ii]]["vx"]

            tracks[ind_current_frame][tracked_id[ii]]["Sigma"] = tracks[
                ind_current_frame - 1
            ][tracked_id[ii]]["Sigma"]
            tracks[ind_current_frame][tracked_id[ii]]["count_fail"] = (
                tracks[ind_current_frame - 1][tracked_id[ii]]["count_fail"] + 1
            )
            tracks[ind_current_frame][tracked_id[ii]]["count"] = (
                tracks[ind_current_frame - 1][tracked_id[ii]]["count"] + 1
            )
            tracks[ind_current_frame][tracked_id[ii]]["num_points"] = tracks[
                ind_current_frame - 1
            ][tracked_id[ii]]["num_points"]
            tracks[ind_current_frame][tracked_id[ii]]["pc_world"] = tracks[
                ind_current_frame - 1
            ][tracked_id[ii]]["pc_world"]

            if self.motion_model == "static":
                tracks[ind_current_frame][tracked_id[ii]]["x"] = tracks[
                    ind_current_frame - 1
                ][tracked_id[ii]]["x"]
            elif self.motion_model == "const_v":
                tracks[ind_current_frame][tracked_id[ii]]["x"] = (
                    tracks[ind_current_frame - 1][tracked_id[ii]]["x"]
                    + tracks[ind_current_frame - 1][tracked_id[ii]]["vx"]
                )
            elif self.motion_model == "measure_only":
                tracks[ind_current_frame][tracked_id[ii]]["x"] = (
                    tracks[ind_current_frame - 1][tracked_id[ii]]["x"]
                    + tracks[ind_current_frame - 1][tracked_id[ii]]["vx"]
                )

            else:
                print("not implemented motion model!!!")


            #compute bounding box
            tracks[ind_current_frame][tracked_id[ii]]["bbox"], angle = compute_bbox(
                    tracks[ind_current_frame][tracked_id[ii]]["pc_world"],
                    city_R_egovehicle,
                    city_t_egovehicle,
                    self.city_name,
                    self.use_map_lane,
                    x_initial=tracks[ind_current_frame][tracked_id[ii]]["x"],
                    fix_size=self.fix_bbox_size)


            # angular velocity estimation is less accurate, 
            #so be conservative here, only use speed and box orientation to update theta
            tracks[ind_current_frame][tracked_id[ii]]["x"][3] = tracks[
                ind_current_frame - 1
            ][tracked_id[ii]]["x"][3]
            tracks[ind_current_frame][tracked_id[ii]]["tracked"] = False

        self.tracks = tracks


    def save_result_label_format(self, lidar_time_stamp, egovehicle_to_city_se3, dataset):
        """
        Save result as input label format
        """
        if dataset == 'Argoverse':
            track_list = []

            for key in self.tracks[self.ind_current_frame].keys():    
                track = {}
                pose = self.tracks[self.ind_current_frame][key]["x"]
                bbox = self.tracks[self.ind_current_frame][key]["bbox"]
                tracked = self.tracks[self.ind_current_frame][key]["tracked"]
    
                theta_local = rotate_orientation(
                    pose, egovehicle_to_city_se3.rotation.transpose()
                )
                pose_local = np.zeros(4)
                pose_local[0:3] = egovehicle_to_city_se3.inverse_transform_point_cloud(pose[0:3][np.newaxis, :])[0]
                pose_local[3] = theta_local
    
                bbox_local = build_bbox(pose_local, bbox.width, bbox.length, bbox.height)
    
                track["center"] = {
                    "x": pose_local[0],
                    "y": pose_local[1],
                    "z": pose_local[2],
                }
                track["rotation"] = {
                    "x": bbox_local.quaternion[1],
                    "y": bbox_local.quaternion[2],
                    "z": bbox_local.quaternion[3],
                    "w": bbox_local.quaternion[0],
                }
                track["length"] = bbox_local.length
                track["width"]  = bbox_local.width
                track["height"] = bbox_local.height
                track["occlusion"] = 0
                track["tracked"] = tracked
                track["timestamp"] = lidar_time_stamp
                track["label_class"] = "VEHICLE"
                track["track_label_uuid"] = key
    
                track_list.append(track)
    
            with open(
                os.path.join(
                    self.path_output, "tracked_object_labels_%s.json" % (lidar_time_stamp)
                ),
                "w",
            ) as outfile:
                json.dump(track_list, outfile, indent=4)

        elif dataset=='KITTI':
            print('TODO ')


        else:
            raise NotImplementedError


    def add_new_track(self, pc_new, track_id_new, city_R_egovehicle, city_t_egovehicle):
        """
        Initailize and add new tracks
        """
        tracks_frame = self.tracks[self.ind_current_frame]
        center = get_polygon_center(pc_new)

        bbox_new, x_initial = initialize_bbox(
            pc_new,
            city_R_egovehicle,
            city_t_egovehicle,
            self.city_name,
            self.use_map_lane,
            self.fix_bbox_size,
        )
        tracks_frame[track_id_new] = {}

        tracks_frame[track_id_new]["center_pc"] = center
        tracks_frame[track_id_new]["x"] = x_initial 
        tracks_frame[track_id_new]["vx"] = np.zeros((4))
        tracks_frame[track_id_new]["pc_world"] = pc_new
        tracks_frame[track_id_new]["tracked"] = True
        tracks_frame[track_id_new]["count"] = 1
        tracks_frame[track_id_new]["Sigma"] = np.eye(4)
        tracks_frame[track_id_new]["Sigma"][3, 3] /= 100  
        tracks_frame[track_id_new]["count_fail"] = 0
        tracks_frame[track_id_new]["bbox"] = bbox_new
        tracks_frame[track_id_new]["count"] = 0
        tracks_frame[track_id_new]["num_points"] = len(pc_new)
        tracks_frame[track_id_new]["tracked"] = True
        print("Add object %s" % track_id_new)

    def save_bev_img(self, dataset_name, log_id, lidar_timestamp, pc, egovehicle_to_city_se3):
        """
        Plot results on bev images and save 
        """
        image_size = 2000
        image_scale = 20
        img = np.zeros((image_size, image_size,3))
        pc = (pc * image_scale)
        pc[:,0] += int(image_size/2)
        pc[:,1] += int(image_size/2)
        pc = pc.astype('int')
    
        ind_valid = np.logical_and(np.logical_and(pc[:,0]>=0 , pc[:,1]>=0), np.logical_and(pc[:,0]<image_size , pc[:,1]<image_size))
        img[pc[ind_valid,0], pc[ind_valid,1],:]=0.4

        path_imgs = os.path.join(self.path_output, 'bev')
        if not os.path.exists(path_imgs):
            os.mkdir(path_imgs)

        for key in self.tracks[self.ind_current_frame].keys():
        
            pose = self.tracks[self.ind_current_frame][key]["x"]
            bbox = self.tracks[self.ind_current_frame][key]["bbox"]
            tracked = self.tracks[self.ind_current_frame][key]["tracked"]
    
            theta_local = rotate_orientation(
                pose, egovehicle_to_city_se3.rotation.transpose()
            )
            pose_local = np.zeros(4)
            pose_local[0:3] = egovehicle_to_city_se3.inverse_transform_point_cloud(pose[0:3][np.newaxis, :])[0]
            pose_local[3] = theta_local

            color_offset = 0.2
            if key not in color_dict.keys():
                color_dict[key] = (min(1,np.random.rand()+color_offset),min(1,np.random.rand()+color_offset),min(1,np.random.rand()+color_offset))
    
            color = (color_dict[key][0] , color_dict[key][1] , color_dict[key][2] )

            w, l, h = bbox.width, bbox.length, bbox.height
            print(w,l,h)
            bbox_2d = np.array([[-l/2, -w/2, 0],[l/2,-w/2, 0],[-l/2,w/2, 0 ],[l/2,w/2, 0]])
            R = np.array([[np.cos(theta_local), -np.sin(theta_local), 0],[np.sin(theta_local), np.cos(theta_local), 0],[0,0,1]])
            bbox_2d = np.matmul(R, bbox_2d.transpose()).transpose()+pose_local[0:3]
            edge_2d = np.array([[0,1], [0,2], [2,3], [1,3]])

            for ii in range(len(edge_2d)):
                p1 = (int(bbox_2d[edge_2d[ii][0],1]*image_scale+image_size/2),int(bbox_2d[edge_2d[ii][0],0]*image_scale+image_size/2))
                p2 = (int(bbox_2d[edge_2d[ii][1],1]*image_scale+image_size/2),int(bbox_2d[edge_2d[ii][1],0]*image_scale+image_size/2))
                cv2.line(img, p1, p2,color = color)

            
        kernel = np.ones((5,5),np.float)
        img = cv2.dilate(img,kernel,iterations = 1)
    
        cv2.putText(img,'%s_%s_%s' % (dataset_name, log_id, lidar_timestamp),
         (100,image_size - 100),
         font,
         fontScale,
         fontColor,
         lineType)

        
        print('Saving img: ', path_imgs)
        cv2.imwrite(os.path.join(path_imgs,'%s_%s_%s.jpg' %  (dataset_name, log_id, lidar_timestamp)), img*255)
    
            
    

    
class Detector:
    """
    Main class for baseline detector
    """

    _RING_CAMERA_LIST = [
        "ring_front_center",
        "ring_front_left",
        "ring_front_right",
        "ring_side_left",
        "ring_side_right",
        "ring_rear_left",
        "ring_rear_right",
    ]

    _CATEGORIES_LIST = [3, 8]  # car, truck
    ground_level = 0  # ground height
    ground_removal_th = 0.3  # margin for ground removal
    use_maskrcnn = True  # use Mask R-CNN to filter detection
    city_name = "MIA"
    min_point_num = 50
    mask_rcnn_detector = None
    dbscan_eps = 1.0
    calib_data = None
    path_debug_output = ""

    img_w = 0
    img_h = 0
    ground_removal_method = "map"
    dataset_name = "Argoverse"

    def __init__(
        self,
        region_type,
        ground_level, 
        ground_removal_th,
        use_maskrcnn,
        city_name,
        min_point_num,
        dbscan_eps,
        calib_data,
        path_debug_output,
        ground_removal_method,
        dataset_name,
        path_rcnn_config="../mask_rcnn/models/e2e_mask_rcnn_R_50_FPN_1x.yaml",

    ):
        self.region_type = region_type
        self.ground_level = ground_level
        self.ground_removal_th = ground_removal_th
        self.use_maskrcnn = use_maskrcnn
        self.city_name = city_name
        self.min_point_num = min_point_num
        self.dbscan_eps = dbscan_eps
        self.calib_data = calib_data
        self.path_debug_output = path_debug_output
        self.ground_removal_method = ground_removal_method
        self.dataset_name = dataset_name

        if use_maskrcnn:
            cfg.merge_from_file(path_rcnn_config)
            cfg.freeze()

            # using setting in official demo
            coco_demo_detector = COCODemo(
                cfg,
                confidence_threshold=0.7,
                min_image_size=224,
            )

            self.mask_rcnn_detector = coco_demo_detector

    def get_candidate_segments(
        self, pc_raw, list_images, id_frame=0, save_results=False
    ):
        """
        Perform detection and return point cloud segments of detected objects
        """
        if self.use_maskrcnn:
            if (self.dataset_name == 'Argoverse'and len(list_images) != len(self._RING_CAMERA_LIST)) or \
            (self.dataset_name == 'KITTI'and len(list_images) <1):
                print("Detector: missing image!!")
                return []

        #Remove ground points
        pc_noground = pc_remove_ground(
            pc_raw,
            self.ground_level,
            ground_removal_th=self.ground_removal_th,
            ground_removal_method=self.ground_removal_method,
        )
        
        if not self.use_maskrcnn:
            pc_segments, tmp = pc_segmentation_dbscan_multilevel(
                pc_noground,
                self.ground_level,
                self.min_point_num,
                eps=self.dbscan_eps,
                ground_removal_th=self.ground_removal_th,
                remove_ground=False,
            )

            return pc_segments

        else:

            masks_rcnn = []
            bboxs_rcnn = []
            self.img_h, self.img_w, c = list_images[0].shape

            if self.dataset_name == 'Argoverse':

                #get detection mask from Mask RCNN
                for ind_cam in range(len(self._RING_CAMERA_LIST)):
                    camera_name = self._RING_CAMERA_LIST[ind_cam]
    
                    img = list_images[ind_cam]
                    top_predictions = self.mask_rcnn_detector.compute_prediction(
                        img
                    )

                    ind_valid = np.isin(
                        top_predictions.get_field("labels").numpy(), self._CATEGORIES_LIST
                    )
    
                    masks = top_predictions.get_field("mask")
                    masks = masks.numpy()[ind_valid]
                    masks_rcnn.append(masks)
                    bbox = top_predictions.bbox.numpy()
                    bboxs_rcnn.append(bbox[ind_valid])
        
                points_valid = np.full(len(pc_noground), False)
                lidar_points_h = point_cloud_to_homogeneous(
                    copy.deepcopy(np.array(pc_noground))
                ).T
                valid_pt_indices = []
    
                # select the points whose projections are inside mask rcnn detection mask
                for ind_cam in range(len(self._RING_CAMERA_LIST)):
    
                    camera_name = self._RING_CAMERA_LIST[ind_cam]
    
                    uv, uv_cam, valid_pts_bool = project_lidar_to_img(
                        lidar_points_h, self.calib_data, camera_name)
                    #)
                    uv_valid = uv[valid_pts_bool, :].astype("int")
    
                    for iii in range(len(masks_rcnn[ind_cam])):
                        mask = masks_rcnn[ind_cam][iii, 0]
                        points_valid[valid_pts_bool] = np.logical_or(
                            points_valid[valid_pts_bool],
                            mask[uv_valid[:, 1], uv_valid[:, 0]] == 1,
                        )
            elif self.dataset_name == 'KITTI':
                camera_name = 'cam_2' # only use cam_2 for KITTI
                for ind_cam in range(len(list_images)):
                        
                    img = list_images[ind_cam]
    
                    top_predictions = self.mask_rcnn_detector.compute_prediction(
                        img)

                    ind_valid = np.isin(
                        top_predictions.get_field("labels").numpy(), self._CATEGORIES_LIST
                    )
    
                    masks = top_predictions.get_field("mask")
                    masks = masks.numpy()[ind_valid]
                    masks_rcnn.append(masks)
                    bbox = top_predictions.bbox.numpy()
                    bboxs_rcnn.append(bbox[ind_valid])
        
                points_valid = np.full(len(pc_noground), False)
                lidar_points_h = point_cloud_to_homogeneous(
                    copy.deepcopy(np.array(pc_noground))
                ).T
                valid_pt_indices = []
    
                # select the points whose projections are inside mask rcnn detection mask
                for ind_cam in range(len(list_images)):
                    uv, uv_cam, valid_pts_bool = project_lidar_to_img_kitti(lidar_points_h, self.calib_data, self.img_h, self.img_w)
    
                
                    uv_valid = uv[valid_pts_bool, :].astype("int")
    
                    for iii in range(len(masks_rcnn[ind_cam])):
                        mask = masks_rcnn[ind_cam][iii, 0]
                        points_valid[valid_pts_bool] = np.logical_or(
                            points_valid[valid_pts_bool],
                            mask[uv_valid[:, 1], uv_valid[:, 0]] == 1,
                        )                


            else:
                raise NotImplementedError

            # remove redundant points(points outside MaskRCNN masks) from pc_noground and do segmentation again
            pc_filtered = pc_noground[points_valid]
            pc_segments, tmp = pc_segmentation_dbscan_multilevel(
                pc_filtered,
                self.ground_level,
                self.min_point_num,
                eps=self.dbscan_eps,
                remove_ground=False,
            )

            # compute average depth of each detected object
            bbox_2D = [None] * len(self._RING_CAMERA_LIST)
            avg_depth = np.zeros((len(self._RING_CAMERA_LIST), len(pc_segments)))
            save_debug_img = True

            if self.dataset_name == "Argoverse":
                num_imgs = len(self._RING_CAMERA_LIST)
            else:
                num_imgs = 1 #KITTI 
            
            for ind_cam in range(num_imgs):
                if self.dataset_name == "Argoverse":
                    camera_name = self._RING_CAMERA_LIST[ind_cam]
    
                img = list_images[ind_cam]
                for ind_seg in range(len(pc_segments)):
                    
                    lidar_points_h = point_cloud_to_homogeneous(copy.deepcopy(pc_segments[ind_seg])).T

                    valid_pt_indices = []
                    if self.dataset_name == "Argoverse":
                        uv, uv_cam, valid_pts_bool = project_lidar_to_img(
                            lidar_points_h,
                            self.calib_data,
                            camera_name,
                            self.img_h,
                            self.img_w,
                        )
                    else:
                        uv,uv_cam, valid_pts_bool = project_lidar_to_img_kitti(
                            lidar_points_h,
                            self.calib_data,
                            self.img_h,
                            self.img_w,
                        )                        
    
                    if valid_pts_bool.sum() == 0:
                        continue
    
                    uv_valid = uv[valid_pts_bool, :].astype("int")
                    avg_depth[ind_cam, ind_seg] = (uv_cam[2, valid_pts_bool].sum() / uv_cam[2, valid_pts_bool].size)
                       
            is_car_maskrcnn = [False] * len(pc_segments)
            used_maskrcnn = [None] *num_imgs

            for ind_cam in range(num_imgs):
                used_maskrcnn[ind_cam] = [False] * len(bboxs_rcnn[ind_cam])
             
            #Remove objects that doesn't satisfy our criteria for car shape
            for ii in range(len(pc_segments)):
                lidar_points_h = point_cloud_to_homogeneous(
                    copy.deepcopy(np.array(pc_segments[ii]))).T
                valid_pt_indices = []

                for ind_cam in range(num_imgs):
                    if self.dataset_name == "Argoverse":
                        camera_name = self._RING_CAMERA_LIST[ind_cam]
    
                    if used_maskrcnn[ind_cam] == None:
                        used_maskrcnn[ind_cam] = [False] * len(bboxs_rcnn[ind_cam])

                    if avg_depth[ind_cam, ii] == 0:
                        continue

                    if self.dataset_name == "Argoverse":
                        uv, uv_cam, valid_pts_bool = project_lidar_to_img(
                            lidar_points_h,
                            self.calib_data,
                            camera_name,
                            self.img_h,
                            self.img_w,
                        )
                    else:
                        uv, uv_cam, valid_pts_bool = project_lidar_to_img_kitti(
                            lidar_points_h,
                            self.calib_data,
                            self.img_h,
                            self.img_w,
                        )                        
    
                    if valid_pts_bool.sum() == 0:
                        continue

                    uv_valid = uv[valid_pts_bool, :].astype("int")

                    for iii in range(len(masks_rcnn[ind_cam])):
                        mask = masks_rcnn[ind_cam][iii, 0]

                        car_ratio = mask[uv_valid[:, 1], uv_valid[:, 0]].sum() / len(
                            uv_valid
                        )

                        if car_ratio > 0.5:
                            used_maskrcnn[ind_cam][iii] = True
                            is_car_maskrcnn[ii] = True

            # keep only the segments inside mask rcnn region whose shape looks like a car
            is_car_maskrcnn = np.array(is_car_maskrcnn, dtype=bool)
            pc_segments = [i for (i, v) in zip(pc_segments, is_car_maskrcnn) if v]

            # add missing segments from mask rcnn
            pc_segments_from2D = []

            for ind_cam in range(num_imgs):
                if self.dataset_name == "Argoverse":
                    camera_name = self._RING_CAMERA_LIST[ind_cam]
    
                lidar_points_h = point_cloud_to_homogeneous(
                    copy.deepcopy(pc_noground)
                ).T
                valid_pt_indices = []
                if self.dataset_name == "Argoverse":
                    uv, uv_cam, valid_pts_bool = project_lidar_to_img(
                        lidar_points_h,
                        self.calib_data,
                        camera_name,
                        self.img_h,
                        self.img_w,
                    )
                else:
                    uv, uv_cam,valid_pts_bool = project_lidar_to_img_kitti(
                        lidar_points_h,
                        self.calib_data,
                        self.img_h,
                        self.img_w,
                    )                        
    
                uv_valid = uv[valid_pts_bool, :].astype("int")
                if valid_pts_bool.sum() == 0:
                    continue

                for iii in range(len(masks_rcnn[ind_cam])):

                    # this segment is already detected by 3D segmentation
                    if used_maskrcnn[ind_cam][iii] == True:
                        continue

                    mask = masks_rcnn[ind_cam][iii, 0]
                    uv_inside_valid = mask[uv_valid[:, 1], uv_valid[:, 0]] == 1
                    pc_cand = pc_noground[valid_pts_bool][uv_inside_valid]

                    if len(pc_cand) == 0:
                        continue
                    if not is_car(pc_cand, self.min_point_num):
                        continue

                    pc_segments_from2D.append(pc_cand)

            if len(pc_segments_from2D) > 0:
                pc_segments = pc_segments + pc_segments_from2D

            pc_segments = sort_pc_segments_by_area(pc_segments)

            # merge repetitive segments
            # compare each pair and assign new segments list indexed by obj_ind
            if len(pc_segments) > 1:
                for ind_loop in range(2):

                    # pc_segments = np.array(pc_segments)
                    assigned_ind = np.full((len(pc_segments)), None)
                    obj_ind = 0

                    for iiii in range(len(pc_segments)):
                        if assigned_ind[iiii] != None:  # object assigned
                            continue

                        assigned_ind[iiii] = obj_ind

                        for jjjj in range(iiii + 1, len(pc_segments)):
                            if assigned_ind[jjjj] != None:  # object assigned
                                continue

                            overlap, pc_merged = check_pc_overlap(
                                pc_segments[iiii], pc_segments[jjjj], self.min_point_num
                            )

                            if overlap:
                                assigned_ind[jjjj] = obj_ind
                                pc_segments[iiii] = pc_merged

                        obj_ind += 1

                    pc_segments = merge_pcs(pc_segments, assigned_ind)

            if save_results:
                for ind_cam in range(num_imgs):
                    if self.dataset_name == "Argoverse":
                        camera_name = self._RING_CAMERA_LIST[ind_cam]
    
                    img = list_images[ind_cam]

                    for ii in range(len(pc_segments)):

                        lidar_points_h = point_cloud_to_homogeneous(
                            copy.deepcopy(np.array(pc_segments[ii]))
                        ).T
                        #valid_pt_indices = []

                        if self.dataset_name == "Argoverse":
                            uv, uv_cam, valid_pts_bool = project_lidar_to_img(
                                lidar_points_h,
                                self.calib_data,
                                camera_name,
                                self.img_h,
                                self.img_w,
                            )
                        else:
                            uv, uv_cam, valid_pts_bool = project_lidar_to_img_kitti(
                                lidar_points_h,
                                self.calib_data,
                                self.img_h,
                                self.img_w,
                            )  
        
                        if valid_pts_bool.sum() == 0:
                            continue

                        uv_valid = uv[valid_pts_bool, :].astype("int")

                        for offset1 in range(-1, 2):
                            for offset2 in range(-1, 2):
                                img[
                                    uv_valid[:, 1] + offset1,
                                    uv_valid[:, 0] + offset2,
                                    :,
                                ] = (
                                    get_color(uv_cam[2, valid_pts_bool], ratio=1)[
                                        :, 0:3
                                    ]
                                    * 255
                                )

                        bbox = [
                            uv_valid[:, 0].min(),
                            uv_valid[:, 0].max(),
                            uv_valid[:, 1].min(),
                            uv_valid[:, 1].max(),
                        ]
                        top_left, bottom_right = [bbox[0], bbox[2]], [bbox[1], bbox[3]]
                        img = cv2.rectangle(
                            img,
                            tuple(top_left),
                            tuple(bottom_right),
                            tuple(fontColor),
                            3,
                        )

                    img = cv2.resize(img,(int( self.img_w/2),int(self.img_h/2)))
                    cv2.imwrite(
                        os.path.join(
                            self.path_debug_output,
                            f"%05d_{camera_name}_vis_after.jpg" % id_frame,
                        ),
                        img,
                    )

        return pc_segments
