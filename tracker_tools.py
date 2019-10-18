
import scipy.spatial.distance
import time
import numpy as np
import sys
import matplotlib.pyplot as plt
import math
import smallestenclosingcircle
import copy
import random
import uuid

from pyquaternion import Quaternion
from argoverse.utils.json_utils import read_json_file
from argoverse.utils.transform import quat2rotmat
from argoverse.utils.ply_loader import load_ply
from argoverse.utils.se3 import SE3
from argoverse.data_loading.object_label_record import ObjectLabelRecord
from argoverse.utils.calibration import project_lidar_to_img
from sklearn.cluster import DBSCAN
from MinimumBoundingBox import MinimumBoundingBox
from shapely.geometry import Polygon
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.synchronization_database import SynchronizationDB
from icp import run_icp
from operator import itemgetter, attrgetter


avm = ArgoverseMap()
idx_bbox = np.array([6, 2, 3, 7])
idx_bbox = idx_bbox.astype("int")
CAR_TURNING_RADIUS = 6
VELOCITY_TH = 0.4
eps = 1e-6


def pc_segmentation_dbscan_multilevel(
    pc_raw,
    ground_level,
    min_point_num,
    eps=1.0,
    ground_removal_th=0.25,
    remove_ground=True,
):
    """
    Do multi-level point cloud segmentation using DBSCAN
    """
    pc_segs = []

    pc_segs_ini, pc = pc_segmentation_dbscan(
        pc_raw,
        ground_level,
        min_point_num,
        eps=eps,
        no_filter=True,
        ground_removal_th=ground_removal_th,
        remove_ground=remove_ground,
    )

    for i in range(len(pc_segs_ini)):
        if len(pc_segs_ini[i]) > 6000:  # try resegment with smaller parameters
            pc_segs_out, pc_seg_large = pc_segmentation_dbscan(
                pc_segs_ini[i],
                ground_level,
                min_point_num,
                eps=0.3,
                no_filter=False,
                remove_ground=False,
            )

            if len(pc_segs_out) > 1:
                pc_segs += pc_segs_out

            elif len(pc_segs_out) == 1:
                if is_car(pc_segs_ini[i], min_point_num):
                    pc_segs.append(pc_segs_ini[i])

        else:

            if is_car(pc_segs_ini[i], min_point_num):
                pc_segs.append(pc_segs_ini[i])

    return pc_segs, pc


def pc_segmentation_dbscan(
    pc_raw,
    ground_level,
    min_point_num,
    no_filter=False,
    eps=2.0,
    leaf_size=30,
    remove_ground=True,
    ground_removal_th=0.25,
):
    """
    Do point cloud segmentation using DBSCAN
    """

    if remove_ground:
        pc = pc_remove_ground(pc_raw, ground_level, ground_removal_th=ground_removal_th)
    else:
        pc = pc_raw

    if len(pc) == 0:
        return [], pc

    db = DBSCAN(
        eps=eps,
        metric="euclidean",
        min_samples=min_point_num,
        n_jobs=4,
        leaf_size=leaf_size,
    ).fit(pc)

    num_seg = db.labels_.max() + 1
    pc_segs = [None] * num_seg
    visited = [None] * num_seg

    for i in range(len(pc)):
        label = db.labels_[i]
        if label == -1:
            continue
        if visited[label] == None:
            pc_segs[label] = pc[i, :][:, np.newaxis].transpose()
            visited[label] = 1

        else:
            pc_segs[label] = np.concatenate(
                (pc_segs[label], pc[i, :][:, np.newaxis].transpose()), axis=0
            )

    ind_keep = []
    for i in range(len(pc_segs)):
        if is_car(pc_segs[i], min_point_num) or no_filter:
            ind_keep.append(i)

    if len(pc_segs) >= 1:
        pc_segs = [pc_segs[i] for i in ind_keep]
    else:
        pc_segs = np.array([])

    return pc_segs, pc


def pc_remove_ground(
    pc_np, ground_range, ground_removal_th=0.25, ground_removal_method="map"
):
    """
    Remove ground points
    """

    if ground_removal_method == "plane_fitting":
        view_range = 100  # only process point cloud within 100m
        inds = (
            (pc_np[:, 0] < view_range)
            & (pc_np[:, 1] < view_range)
            & (pc_np[:, 0] > -view_range)
            & (pc_np[:, 1] > -view_range)
        )

        pc_np = pc_np[inds]
        plane_range_th = 1000
        pc_ground = pc_np[
            (pc_np[:, 0] < plane_range_th)
            & (pc_np[:, 1] < plane_range_th)
            & (pc_np[:, 2] < ground_range)
            & (pc_np[:, 0] > -plane_range_th)
            & (pc_np[:, 1] > -plane_range_th),
            :,
        ]

        is_p, normal, d = is_plane(pc_ground[::10, :], 0.3)
        if normal[2] < 0:
            normal = -normal
            d = -d

        car_height_max = 4.0
        errors = np.matmul(pc_np, normal) + d
        pc_np = pc_np[(errors > ground_removal_th) & (errors < car_height_max), :]


    elif ground_removal_method == "threshold":
        view_range = 300
        inds = (
            (pc_np[:, 0] < view_range)
            & (pc_np[:, 1] < view_range)
            & (pc_np[:, 2] > ground_range)
            & (pc_np[:, 0] > -view_range)
            & (pc_np[:, 1] > -view_range)
        )

        pc_np = pc_np[inds]
    elif ground_removal_method == "map":
        # already done together with roi extraction
        return pc_np
    else:
        print("ground removal method not implemented!!")

    return pc_np




def is_plane(pc, th):
    """
    Do plane detection
    """
    sample_size = 100
    if len(pc) > sample_size:
        random.sample(np.arange(len(pc)).tolist(), sample_size)

    center = pc.sum(axis=0) / pc.shape[0]
    # run SVD
    u, s, vh = np.linalg.svd(pc - center)

    # unitary normal vector
    u_norm = vh[2, :]
    d = -np.dot(u_norm, center)
    errors = np.matmul(pc, u_norm) + d
    errors = (errors ** 2).sum() / len(pc)

    return errors < th, u_norm, d


def get_color(value, ratio=0.1, ratio2=1):
    """
    Transform depth value to 3 channel color
    """

    cmap = plt.get_cmap("viridis_r")
    return cmap(ratio2 * np.log(value * ratio))



def point_cloud_to_homogeneous(points):
    """
    Append ones to point cloud array to make it homogeneous
    """
    num_pts = points.shape[0]
    return np.hstack([points, np.ones((num_pts, 1))])



def initialize_bbox(pc, R_world, t_world, city_name, use_map_lane, fix_bbox_size):
    """
    Convert input point cloud to tracker bounding box
    """
    return compute_bbox(
        pc, R_world, t_world, city_name, use_map_lane, fix_size=fix_bbox_size
    )



def get_polygon_center(pc):
    """
    Estimate object center 
    """
    sample_size = 100
    if len(pc) > sample_size:
        random.sample(np.arange(len(pc)).tolist(), sample_size)

    pc = np.array(pc)
    center = np.sum(pc, axis=0) / len(pc)
    circle = smallestenclosingcircle.make_circle(pc[:, 0:2])

    return np.array([circle[0], circle[1], center[2]])



def get_icp_measurement(pc_in_raw, pc_out_raw, do_exhaustive_serach=True):
    """
    Perform ICP and return transformation matrix, fitness score, and accumulated point cloud
    """
    if len(pc_in_raw) <= 20 or len(pc_out_raw) <= 20:
        print("too few points for icp!!!")
        return np.eye(4), 0, pc_out_raw, 2

    num_in = len(pc_in_raw)
    num_out = len(pc_out_raw)

    center_in = (np.sum(pc_in_raw, axis=0) / num_in)[:, np.newaxis]
    pc_in = pc_in_raw - center_in.transpose()
    pc_out = pc_out_raw - center_in.transpose()

    #select only 500 points
    num_select = 500
    ind_in = np.random.choice(num_in, min(num_select, num_in))
    ind_out = np.random.choice(num_out, min(num_select, num_out))

    pc_in = pc_in[ind_in]
    pc_out = pc_out[ind_out]

    pc_in[:, 2] = 0  # heuristicly using 2D points
    pc_out[:, 2] = 0


    if do_exhaustive_serach:
        delta_theta_z = [0]  
        delta_x = np.arange(-0.5, 0.5001, 0.5)  
        delta_y = np.arange(-0.5, 0.5001, 0.5)  
    else:
        delta_theta_z = [0]
        delta_x = [0]
        delta_y = [0]

    num_total = len(delta_theta_z) * len(delta_x) * len(delta_y)
    results = []
    data_input = []
    for iter_t in range(len(delta_theta_z)):
        for iter_x in range(len(delta_x)):
            for iter_y in range(len(delta_y)):
                data = (
                    delta_theta_z,
                    delta_x,
                    delta_y,
                    pc_in,
                    pc_out,
                    iter_t,
                    iter_x,
                    iter_y,
                )
                results.append(run_icp(data))

    transfs = [None] * num_total
    fitness = [None] * num_total
    for i in range(num_total):
        transfs[i] = results[i][0]
        fitness[i] = results[i][1]

    fitness = np.array(fitness)
    transf_best = transfs[np.argmin(fitness)]
    transf_se3 = SE3(rotation=transf_best[0:3, 0:3], translation=transf_best[0:3, 3])
    pc_aligned = transf_se3.transform_point_cloud(pc_in)
    pc_accu = np.concatenate((pc_aligned, pc_out), axis=0) + center_in.transpose()
    bbox_min = MinimumBoundingBox.MinimumBoundingBox(pc_in[:, 0:2])

    return transf_best, fitness.min()/(bbox_min.area/15), pc_accu


def compute_bbox(
    pc, R_world, t_world, city_name, use_map_lane, fix_size=True, x_initial=[]
):
    """
    Compute bounding box from point cloud segment

    """

    bbox_min = MinimumBoundingBox.MinimumBoundingBox(pc[:, 0:2])
    l01_smallest = bbox_min.length_parallel
    l02_smallest = bbox_min.length_orthogonal

    if l02_smallest < l01_smallest:
        angle = bbox_min.unit_vector_angle

        unit_vector_vertical = (bbox_min.unit_vector[1], -bbox_min.unit_vector[0])

        if (
            np.cross(
                [unit_vector_vertical[0], unit_vector_vertical[1], 0],
                [bbox_min.unit_vector[0], bbox_min.unit_vector[1], 0],
            )[2]
            < 0
        ):
            unit_vector_vertical = -unit_vector_vertical

        vec_parllel = np.array(bbox_min.unit_vector)
        vec_vertical = np.array(unit_vector_vertical)

    else:

        tmp = l01_smallest
        l01_smallest = l02_smallest
        l02_smallest = tmp
        angle = bbox_min.unit_vector_angle + np.pi / 2

        if angle > np.pi:
            angle -= 2 * np.pi
        unit_vector_vertical = (bbox_min.unit_vector[1], -bbox_min.unit_vector[0])
        if (
            np.cross(
                [unit_vector_vertical[0], unit_vector_vertical[1], 0],
                [bbox_min.unit_vector[0], bbox_min.unit_vector[1], 0],
            )[2]
            < 0
        ):
            unit_vector_vertical = -unit_vector_vertical

        vec_vertical = np.array(bbox_min.unit_vector)
        vec_parllel = np.array(unit_vector_vertical)

    center = bbox_min.rectangle_center

    width = l02_smallest  
    length = l01_smallest 

    h_bottom = pc[:, 2].min() - 0.3
    height = pc[:, 2].max() - h_bottom

    x_initial = x_initial.copy()
    if len(x_initial) == 0:
        z_avg = pc[:, 2].sum() / len(pc)
        x_initial = np.array([center[0], center[1], z_avg, angle])

    if use_map_lane:
        lane_dir_vector, confidence = get_lane_direction_api(
            np.array([center[0], center[1]]), R_world, t_world, city_name
        )

        if confidence > 0.5:
            x_initial[3] = np.arctan2(lane_dir_vector[1], lane_dir_vector[0])

    else:
        # if not using map lane, use angle estimated by tightest bounding box
        x_initial[3] = angle

    if  fix_size:
        length = 4.5
        width = 1.8

    bbox = build_bbox(x_initial, width, length, height)

    return bbox, x_initial


def build_bbox(pose, width, length, height):
    """
    Convert bounding box to label format 

    """
    R = np.array(
        [
            [np.cos(pose[3]), -np.sin(pose[3]), 0],
            [np.sin(pose[3]), np.cos(pose[3]), 0],
            [0, 0, 1],
        ]
    )
    q = Quaternion(matrix=R)
    return ObjectLabelRecord(
        quaternion=[q.scalar, q.vector[0], q.vector[1], q.vector[2]],
        translation=pose[0:3].copy(),
        length=length,
        width=width,
        height=height,
        occlusion=0,
    )


def generate_ID():
    """
    Generate tracking IDs to identify objects
    """
    return str(uuid.uuid1())


def vector_proj(v_input, v_target):
    """
    Do vector projection on another vector
    """
    v_target /= np.linalg.norm(v_target[0:2, 0])
    return np.dot(v_input[:, 0], v_target[:, 0]) * v_target


def check_orientation(v, theta):
    """
    Make sure orientation theta and velocity point to the same direction

    """

    v_theta = np.array([np.cos(theta), np.sin(theta)])

    if np.dot(v[0:2], v_theta) < 0:
        theta = np.pi + theta

    # theta is in -pi~pi
    if theta > math.pi:
        theta = theta - 2 * math.pi

    elif theta < -math.pi:
        theta = 2 * math.pi + theta

    return theta


def do_tracking(
    x_prev,
    v_prev,
    Sigma_prev,
    z_detect,
    theta_conf_detect,
    z_icp,
    icp_fitness,
    pc_segment,
    motion_model="static",
    measurement_model="detect",
):
    """
    Do tracking by merging measurement and motion model
    """

    # handle 2 sources of motion measurements : 1. detection 2.icp
    C_detect = np.diag([0.1, 0.1, 0.0000001, 0.01 / max(theta_conf_detect, 0.5)])
    C_icp = np.diag([0.1, 0.1, 0.1, 0.1]) * min(1 / 10 * np.exp(icp_fitness * 100), 10)

    # when too few points are avaliable, detect/icp results are not robust
    if len(pc_segment) < 50:
        C_detect *= 10
        C_icp *= 10

    # align theta orientation, assuming cars don't turn 180 degree within one frame
    diff_theta_icp = x_prev[3] - z_icp[3]
    if diff_theta_icp > np.pi / 2:
        z_icp[3] += np.pi
    elif diff_theta_icp < -np.pi / 2:
        z_icp[3] -= np.pi

    diff_theta_detect = x_prev[3] - z_detect[3]
    if diff_theta_detect > np.pi / 2:
        z_detect[3] += np.pi
    elif diff_theta_detect < -np.pi / 2:
        z_detect[3] -= np.pi

    if measurement_model == "both":
        C_detect_inv = np.linalg.inv(C_detect)
        C_icp_inv = np.linalg.inv(C_icp)

        # compute measurements by fusing icp and detection
        C_measurement = np.linalg.inv(C_detect_inv + C_icp_inv)
        z = np.matmul(
            C_measurement,
            np.matmul(C_detect_inv, z_detect[:, np.newaxis])
            + np.matmul(C_icp_inv, z_icp[:, np.newaxis]),
        )

    elif measurement_model == "icp":
        C_measurement = C_icp
        z = z_icp[:, np.newaxis]

    elif measurement_model == "detect":
        C_measurement = C_detect
        z = z_detect[:, np.newaxis]
    else:
        print("measurement model not implemented!!")

    #Apply three types of motion model
    if motion_model == "static":
        x_pred = x_prev
    elif motion_model == "const_v":
        x_pred = x_prev + v_prev
    elif motion_model == "measure_only":
        return z[:, 0], C_measurement
    else:
        print("motion model not implemented!")

    # use the velocity direction as motion model theta
    C_motion = np.diag([0.1, 0.1, 0.1, 0.01])
    VELOCITY_TH_MOTION=2

    if np.linalg.norm(v_prev[0:2]) < VELOCITY_TH_MOTION:
        x_pred[3] = x_prev[3]
        C_motion[3, 3] *= np.linalg.norm(v_prev[0:2]) * 0.5 + eps
    else:
        C_motion[3, 3] *= VELOCITY_TH_MOTION/np.linalg.norm(v_prev[0:2])
        x_pred[3] = np.arctan2(v_prev[1], v_prev[0])


    # Simplified kalman filter equations
    I = np.eye(4)
    Sigma_pred = Sigma_prev + C_motion
    K = np.matmul(Sigma_pred, np.linalg.inv(Sigma_pred + C_measurement))
    x_est = x_pred[:, np.newaxis] + np.matmul(K, z - x_pred[:, np.newaxis])
    Sigma_est = np.matmul((I - K), Sigma_pred)

    # theta is in -pi~pi
    if x_est[3, 0] > math.pi:
        x_est[3, 0] = x_est[3, 0] - 2 * math.pi

    elif x_est[3, 0] < -math.pi:
        x_est[3, 0] = 2 * math.pi + x_est[3, 0]

    return x_est[:, 0], Sigma_est



def show_pc_segments(pc_all, pc_all2, pc_segments):
   """
   3D visualization of segmentation result
   """
   from mayavi import mlab
   mlab.figure(bgcolor=(0.0,0.0,0.0))
   mlab.points3d(pc_all[:,0],pc_all[:,1],pc_all[:,2],scale_factor=0.1,color=(0.3,0.2,0.2),opacity=0.3)
   mlab.points3d(pc_all2[:,0],pc_all2[:,1],pc_all2[:,2],scale_factor=0.1,color=(0.4,0.4,0.4),opacity=1.0)

   for i in range(len(pc_segments)):
       color = (random.random()*0.8 + 0.2  ,random.random()*0.8 + 0.2  ,random.random()*0.8 + 0.2)
       nodes1 = mlab.points3d(pc_segments[i][:,0],pc_segments[i][:,1],pc_segments[i][:,2],scale_factor=0.1,color=color,opacity=0.8)

       nodes1.glyph.scale_mode = 'scale_by_vector'


   length_axis = 2.0
   linewidth_axis = 0.1
   color_axis = (1.0,0.0,0.0)
   pc_axis = np.array([[0,0,0], [length_axis,0,0], [0,length_axis,0],[0,0,length_axis]])
   mlab.plot3d(pc_axis[0:2,0], pc_axis[0:2,1], pc_axis[0:2,2],tube_radius=linewidth_axis, color=(1.0,0.0,0.0))
   mlab.plot3d(pc_axis[[0,2],0], pc_axis[[0,2],1], pc_axis[[0,2],2],tube_radius=linewidth_axis, color=(0.0,1.0,0.0))
   mlab.plot3d(pc_axis[[0,3],0], pc_axis[[0,3],1], pc_axis[[0,3],2],tube_radius=linewidth_axis, color=(0.0,0.0,1.0))


   mlab.show()


def is_car(pc, min_point_num):
    """
    Shape heuristics for car detection
    """

    pc = np.array(pc)
    if len(pc) < min_point_num:
        return False

    height = pc[:, 2].max() - pc[:, 2].min()
    if height < 0.5:
        return False

    num_lower = len(pc[pc[:, 2] < (height / 2 + pc[:, 2].min()), :])
    if num_lower / len(pc) > 0.9:
        return False

    bbox_min = MinimumBoundingBox.MinimumBoundingBox(pc[:, 0:2])
    l01 = bbox_min.length_parallel
    l02 = bbox_min.length_orthogonal
    area = l01 * l02

    if (
        (area < 1.5 or area > 20)
        or ((l01 > 6 or l02 > 6))
        or ((l01 < 0.4 and l02 < 0.4))
    ):
        return False

    return True



def sort_pc_segments_by_area(pc_segments):
    """
    Sort a list of point cloud segments by bounding box area
    """
    areas = []
    for i in range(len(pc_segments)):

        b1 = get_pc_bbox(pc_segments[i])
        b1_c = Polygon(b1)

        areas.append(b1_c.area)

    return [p for _, p in sorted(zip(areas, pc_segments), key=itemgetter(0))]


def get_pc_bbox(pc):
    """
    Compute bounding box
    """
    return [
        (pc[:, 0].min(), pc[:, 1].min()),
        (pc[:, 0].max(), pc[:, 1].min()),
        (pc[:, 0].max(), pc[:, 1].max()),
        (pc[:, 0].min(), pc[:, 1].max()),
    ]


def check_pc_overlap(pc1, pc2, min_point_num):
    """
    Check if the bounding boxes of the 2 given point clouds overlap
    """
    b1 = get_pc_bbox(pc1)
    b2 = get_pc_bbox(pc2)

    b1_c = Polygon(b1)
    b2_c = Polygon(b2)
    inter_area = b1_c.intersection(b2_c).area
    union_area = b1_c.area + b2_c.area - inter_area

    if b1_c.area > 11 and b2_c.area > 11:
        overlap = (inter_area / union_area) > 0.5

    elif inter_area > 0:
        overlap = True  
    else:
        overlap = False

    pc_merged = pc2
    if overlap:

        bbox_min = MinimumBoundingBox.MinimumBoundingBox(
            np.concatenate((pc1[:, 0:2], pc2[:, 0:2]), axis=0)
        )
        l01 = bbox_min.length_parallel
        l02 = bbox_min.length_orthogonal
        area = l01 * l02

        # shape doesn't look like car bbox
        if ((area < 2 or area > 12)
            or ((l01 > 4.6 or l02 > 4.6))
            or ((l01 < 1 or l02 < 1))
            or union_area > 15
        ):
            if b1_c.area > b2_c.area:
                pc_merged = pc1
            else:
                pc_merged = pc2
        else:

            idx_overlap = np.zeros((len(pc1)))
            for i in range(len(pc1)):
                diff = pc2 - pc1[i]
                diff = np.sum(diff ** 2, axis=1)
                if 0 in diff:
                    idx_overlap[i] = 1

            pc_merged = np.concatenate((pc_merged, pc1[idx_overlap == 0]), axis=0)

    if not is_car(pc_merged, min_point_num):
        overlap = False

    return overlap, pc_merged


def merge_pcs(pcs, inds):
    """
    Merge point cloud segments
    """

    pcs = np.array(pcs)
    pcs_out = []
    for i in range(inds.max() + 1):
        inds_selected = inds == i  # overlapping pcs are labeld with same ind

        pcs_selected = pcs[inds_selected]

        pc = pcs_selected[0]
        for ii in range(1, len(pcs_selected)):
            pc = merge_pc(pcs_selected[ii], pc)

        pcs_out.append(pc)

    return pcs_out


def check_pc_bbox_overlap(pc1, bbox2):
    """
    Check if a point cloud overlaps with a bounding box
    """
    bbox = bbox2.as_3d_bbox()  
    b2_c = Polygon(bbox[idx_bbox, :])
    b1 = get_pc_bbox(pc1)
    b1_c = Polygon(b1)
    inter_area = b1_c.intersection(b2_c).area

    overlap = inter_area > 0

    return overlap


def transform(x_in, R, t, transpose=True, inverse=False):
    """
    Do spatial transformation for 3D points 
    """
    if inverse:
        R = R.transpose()
        t = -np.matmul(R, t)
    if len(x_in.shape) == 1:
        x = x_in[:, np.newaxis]  # shape = 1x3
    else:
        x = x_in

        if transpose:
            x = x.transpose()

    if len(t.shape) == 1:
        t = t[:, np.newaxis]

    if len(x_in.shape) == 1:
        return (np.matmul(R, x) + t)[:, 0]
    else:
        if transpose:
            return (np.matmul(R, x) + t).transpose()
        else:
            return np.matmul(R, x) + t


def rotate_orientation(x, rotation):
    """
    convert rotation matrix to orientation angle
    """

    pose_head = np.array([np.cos(x[3]), np.sin(x[3]), 0])[np.newaxis, :]

    transf_se3 = SE3(rotation=rotation[0:3, 0:3], translation=np.zeros((3)))

    pose_head_new = transf_se3.transform_point_cloud(pose_head)

    theta_new = np.arctan2(pose_head_new[0, 1], pose_head_new[0, 0])

    return theta_new

def get_z_icp(x_prev, transf):
    """
    Apply transformation matrix from icp to previous pose to get measurement
    """

    pose_new = x_prev[0:3] + transf[0:3, 3]
    pose_head = np.array([np.cos(x_prev[3]), np.sin(x_prev[3]), 0])[np.newaxis, :]
    transf_se3 = SE3(rotation=transf[0:3, 0:3], translation=np.zeros((3)))
    pose_head_new = transf_se3.transform_point_cloud(pose_head)

    theta_new = np.arctan2(pose_head_new[0, 1], pose_head_new[0, 0])

    # clip rotation angle given turning radius
    dx = pose_new[0] - x_prev[0]
    dy = pose_new[1] - x_prev[1]

    delta_theta = theta_new - x_prev[3]
    theta_th = np.sqrt(dx ** 2 + dy ** 2) / CAR_TURNING_RADIUS
    if delta_theta > 0:
        delta_theta = min(delta_theta, theta_th)
    else:
        delta_theta = max(delta_theta, -theta_th)

    output = x_prev + np.array([0, 0, 0, delta_theta])
    output[0:3] = pose_new[0:3]

    return output


def merge_pc(pc1, pc2):
    """
    Merge two point cloud segments
    """
    idx_overlap = np.zeros((len(pc1)))
    for i in range(len(pc1)):
        diff = pc2 - pc1[i]
        diff = np.sum(diff ** 2, axis=1)
        if 0 in diff:
            idx_overlap[i] = 1

    pc_merged = np.concatenate((pc2, pc1[idx_overlap == 0]), axis=0)

    return pc_merged


def get_z_detect(
    pc_segment, v_prev, use_map_lane, city_R_egovehicle, city_t_egovehicle, city_name
):
    """
    Compute object pose from detected point cloud segment
    """

    pc_center = get_polygon_center(pc_segment)
    bbox_min = MinimumBoundingBox.MinimumBoundingBox(pc_segment[:, 0:2])
    vec_parllel = np.array(bbox_min.unit_vector)

    theta = np.arctan2(vec_parllel[1], vec_parllel[0])
    confidence = bbox_min.area / max(15, bbox_min.area)


    if use_map_lane:
        lane_dir_vector, confidence_lane = get_lane_direction_api(
            np.array([pc_center[0], pc_center[1]]),
            city_R_egovehicle,
            city_t_egovehicle,
            city_name,
        )


        if confidence_lane > confidence:
            theta = np.arctan2(lane_dir_vector[1], lane_dir_vector[0])
            confidence = confidence_lane

    return np.append(pc_center, theta), confidence




def get_lane_direction_api(pc, R_world, t_world, city_name):
    """
    Get lane direction using Argoverse API
    """

    lane_dir_vector, confidence = avm.get_lane_direction(pc, city_name, visualize=False)

    return lane_dir_vector, confidence


def get_pc_inside_bbox(pc_raw, bbox):
    """
    Compute the points inside the given bounding box
    """

    U = []
    V = []
    W = []
    P1 = []
    P2 = []
    P4 = []
    P5 = []
    index = []

    u = bbox[1] - bbox[0]
    v = bbox[2] - bbox[0]
    w = np.zeros((3, 1))  
    w[2, 0] += bbox[3]

    p5 = w + bbox[0]

    U.append(u[0:3, 0])
    V.append(v[0:3, 0])
    W.append(w[0:3, 0])
    P1.append(bbox[0][0:3, 0])
    P2.append(bbox[1][0:3, 0])
    P4.append(bbox[2][0:3, 0])
    P5.append(p5[0:3, 0])


    if len(U) == 0:
        return []

    U = np.array(U)
    W = np.array(W)
    V = np.array(V)
    P1 = np.array(P1)
    P2 = np.array(P2)
    P4 = np.array(P4)
    P5 = np.array(P5)

    dot1 = np.matmul(U, pc_raw.transpose(1, 0))
    dot2 = np.matmul(V, pc_raw.transpose(1, 0))
    dot3 = np.matmul(W, pc_raw.transpose(1, 0))
    u_p1 = np.tile((U * P1).sum(axis=1), (len(pc_raw), 1)).transpose(1, 0)
    v_p1 = np.tile((V * P1).sum(axis=1), (len(pc_raw), 1)).transpose(1, 0)
    w_p1 = np.tile((W * P1).sum(axis=1), (len(pc_raw), 1)).transpose(1, 0)
    u_p2 = np.tile((U * P2).sum(axis=1), (len(pc_raw), 1)).transpose(1, 0)
    v_p4 = np.tile((V * P4).sum(axis=1), (len(pc_raw), 1)).transpose(1, 0)
    w_p5 = np.tile((W * P5).sum(axis=1), (len(pc_raw), 1)).transpose(1, 0)

    flag = np.logical_and(
        np.logical_and(
            in_between_matrix(dot1, u_p1, u_p2), in_between_matrix(dot2, v_p1, v_p4)
        ),
        in_between_matrix(dot3, w_p1, w_p5),
    )

    return pc_raw[flag[0, :]]


def in_between_matrix(x, v1, v2):
    """
    Check is the elements in x is between v1 and v2
    """
    return np.logical_or(
        np.logical_and(x <= v1, x >= v2), np.logical_and(x <= v2, x >= v1)
    )


def leave_only_driveable_region(
    lidar_pts, egovehicle_to_city_se3, ground_removal_method, city_name="MIA"
): 
    """
    Remove points outside driveable region
    """

    drivable_area_pts = copy.deepcopy(lidar_pts)
    drivable_area_pts = egovehicle_to_city_se3.transform_point_cloud(
        drivable_area_pts
    )  # put into city coords

    drivable_area_pts = avm.remove_non_driveable_area_points(
        drivable_area_pts, city_name
    )

    if ground_removal_method == "map":
        drivable_area_pts = avm.remove_ground_surface(drivable_area_pts, city_name)
    drivable_area_pts = egovehicle_to_city_se3.inverse_transform_point_cloud(
        drivable_area_pts
    )  # put back into ego-vehicle coords
    return drivable_area_pts


def leave_only_roi_region(
    lidar_pts, egovehicle_to_city_se3, ground_removal_method, city_name="MIA"
):
    """
    Remove points outside map ROI
    """

    drivable_area_pts = copy.deepcopy(lidar_pts)
    drivable_area_pts = egovehicle_to_city_se3.transform_point_cloud(
        drivable_area_pts
    )  # put into city coords

    drivable_area_pts = avm.remove_non_roi_points(drivable_area_pts, city_name)

    if ground_removal_method == "map":
        drivable_area_pts = avm.remove_ground_surface(drivable_area_pts, city_name)
    drivable_area_pts = egovehicle_to_city_se3.inverse_transform_point_cloud(
        drivable_area_pts
    )  # put back into ego-vehicle coords

    return drivable_area_pts



def project_lidar_to_img_kitti(lidar_points_h, calib_data, img_h, img_w):
    """
    Project KITTI LiDAR on KITTI imagees

    """

    T_velo_to_cam = calib_data['Tr_velo_cam']
    P2 = calib_data['P2']

    uv_cam = np.matmul(T_velo_to_cam, lidar_points_h)
    uv_cam[0,:] /= uv_cam[3,:]
    uv_cam[1,:] /= uv_cam[3,:]
    uv_cam[2,:] /= uv_cam[3,:]
    uv_cam[3,:]=1

    uv = np.matmul(P2, uv_cam)
    uv[0,:] /= uv[2,:]
    uv[1,:] /= uv[2,:]
    uv = uv.transpose()

    ind_valid = (uv[:,0] >= 0) * (uv[:,1] >= 0 ) *  (uv[:,0] <= img_w-1) * (uv[:,1] <= img_h-1 ) * (uv_cam[2,:] > 0 )

    return uv, uv_cam, ind_valid
