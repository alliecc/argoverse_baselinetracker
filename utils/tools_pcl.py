# -*- coding: utf-8 -*-
import numpy as np
import pcl
from open3d import *
import random
import sys

sys.path.append("..")
sys.path.append("../..")
sys.path.append("../bboxmaster")
import utils
from utils import tools
from utils.tools import *
import json

sys.path.append("../utils/bboxmaster")
from pyquaternion import Quaternion
from utils.bboxmaster.bbox.bbox3d import BBox3D
import csv
from numpy import cos, sin
import math
from sklearn.cluster import dbscan
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import glob
from scipy.spatial import ConvexHull

CAR_TURNING_RADIUS = 4
USE_IN_ACCU_AND_EKF = 0
USE_IN_ACCU_BUT_EKF = 1
USE_IN_NOT_ACCU_OR_EKF = 2
max_dist_local = 5


def load_ply(path_file):
    pcd_load = read_point_cloud(path_file)
    xyz_load = np.asarray(pcd_load.points)
    return xyz_load


def save_ply(path_file, pc):
    # import pdb; pdb.set_trace()
    if isinstance(pc, int):
        print("pc has no value")
        return

    if len(pc) < 8:
        print("Too few points for saving ply!!!!")
        return

    pcd = PointCloud()
    pcd.points = Vector3dVector(pc)
    write_point_cloud(path_file, pcd)


def pc_remove_plane(pc, plane_th):

    max_iter = 20
    point_th = 1000
    i = 0

    indices_plane = []
    cloud = pcl.PointCloud(pc.astype(np.float32))
    planes_normal = []
    planes_d = []

    while i < max_iter:

        seg = cloud.make_segmenter_normals(ksearch=50)
        seg.set_optimize_coefficients(True)
        seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
        seg.set_normal_distance_weight(0.1)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_max_iterations(100)
        seg.set_distance_threshold(0.05)
        # seg.set_samples_maxdist(0.2)
        indices, model = seg.segment()
        # if len(indices) < point_th:
        #    break

        plane_pc = cloud.extract(indices, negative=False)
        plane_pc = pcl_to_numpy(plane_pc)
        if len(plane_pc) == 0:
            break
        # import pdb; pdb.set_trace()

        is_p, normal, d = is_plane(plane_pc, 0.3)

        center = np.sum(plane_pc, axis=0) / len(plane_pc)
        dists_local = plane_pc - center
        dists_local = np.sqrt(np.sum(dists_local ** 2, axis=1))

        if (
            max(dists_local) > max_dist_local
            and np.abs(normal[2]) < 0.1
            and plane_pc.size > 2000
        ):

            planes_normal.append(normal)
            planes_d.append(d)

        # print(plane_pc.size)

        # save_ply('test_' +str(i) + '.ply' , plane_pc)
        # is_plane(plane_pc, 0.5)
        #
        cloud = cloud.extract(indices, negative=True)
        i += 1

    # cloud = pcl.PointCloud(pc.astype(np.float32))
    # cloud = cloud.extract(indices_plane, negative=True)
    # import pdb; pdb.set_trace()
    valid_idx = np.ones(len(pc))
    for i in range(len(planes_normal)):
        dists = np.abs(np.matmul(pc, planes_normal[i]) + planes_d[i])
        valid_idx[dists < plane_th] = 0
        # save_ply('test_'+str(i)+'.ply',pc[valid_idx.astype('bool'),:])
        # import pdb; pdb.set_trace()

    return pc[valid_idx.astype("bool"), :]  # pcl_to_numpy(cloud)

    # import pdb; pdb.set_trace()

    # return plane_pc


import random


def is_plane(pc, th):
    sample_size = 100
    if len(pc) > sample_size:
        random.sample(np.arange(len(pc)).tolist(), sample_size)

    center = pc.sum(axis=0) / pc.shape[0]
    # run SVD
    u, s, vh = np.linalg.svd(pc - center)

    # unitary normal vector
    u_norm = vh[2, :]
    d = -np.dot(u_norm, center)
    # import pdb; pdb.set_trace()
    errors = np.matmul(pc, u_norm) + d

    errors = (errors ** 2).sum() / len(pc)

    return errors < th, u_norm, d


def pc_segmentation_dbscan_multilevel(
    pc_raw,
    ground_level,
    min_point_num,
    plane_th,
    local_th,
    eps=1.0,
    ground_removal_th=0.25,
    remove_ground=True,
):
    pc_segs = []
    # import pdb; pdb.set_trace
    pc_segs_ini, pc = pc_segmentation_dbscan(
        pc_raw,
        ground_level,
        min_point_num,
        plane_th,
        local_th,
        eps=eps,
        no_filter=True,
        ground_removal_th=ground_removal_th,
        remove_ground=remove_ground,
    )

    for i in range(len(pc_segs_ini)):
        if len(pc_segs_ini[i]) > 6000:  # try resegment
            pc_segs_out, pc_seg_large = pc_segmentation_dbscan(
                pc_segs_ini[i],
                ground_level,
                min_point_num,
                plane_th,
                local_th,
                eps=0.3,
                no_filter=False,
                remove_ground=False,
            )

            if len(pc_segs_out) > 1:
                pc_segs += pc_segs_out
            elif len(pc_segs_out) == 1:
                # center_seg =  get_polygon_center(pc_segs_ini[i])
                # dists_local = pc_segs_ini[i] - center_seg
                # dists_local = np.sqrt(np.sum(dists_local**2,axis=1))

                if is_car(
                    pc_segs_ini[i], min_point_num
                ):  # max(dists_local) < local_th and is_car(pc_segs_ini[i]):
                    pc_segs.append(pc_segs_ini[i])

            # import pdb; pdb.set_trace()
        else:
            # center_seg = np.sum(pc_segs_ini[i], axis=0)/ len(pc_segs_ini[i])
            # center_seg =  get_polygon_center(pc_segs_ini[i])
            # dists_local = pc_segs_ini[i] - center_seg
            # dists_local = np.sqrt(np.sum(dists_local**2,axis=1))

            if is_car(pc_segs_ini[i], min_point_num):  #  max(dists_local) < local_th:
                pc_segs.append(pc_segs_ini[i])

    return pc_segs, pc


from sklearn.metrics import euclidean_distances
import scipy.spatial.distance
import time


def pc_segmentation_dbscan(
    pc_raw,
    ground_level,
    min_point_num,
    plane_th,
    local_th,
    eps=2.0,
    no_filter=False,
    leaf_size=30,
    remove_ground=True,
    ground_removal_th=0.25,
    remove_plane=False,
):
    # import pdb; pdb.set_trace()
    if remove_ground:
        pc_noground = pc_remove_ground(
            pc_raw, ground_level, ground_removal_th=ground_removal_th
        )

        if remove_plane:
            pc = pc_remove_plane(pc_noground, plane_th)
        else:
            pc = pc_noground
    else:
        if remove_plane:
            pc = pc_remove_plane(pc_raw, plane_th)
        else:
            pc = pc_raw

    # tmp = []
    # tmp.append(pc_raw)
    # tmp.append(pc_noground)
    # tmp.append(pc)

    # pc = pc_remove_plane(pc)
    # db = DBSCAN(eps=3,  min_samples=50, n_jobs=4).fit(pc)
    #
    # num_seg = db.labels_.max()+1
    # pc_segs = [None] * num_seg
    # visited =  [None] * num_seg
    ##
    # for i in range(len(pc)):
    #    label =  db.labels_[i]
    #    if label == -1:
    #        continue
    #    if visited[label] == None:
    #        pc_segs[label] = pc[i,:][:,np.newaxis].transpose()
    #        visited[label] = 1
    #
    #    else:
    #        pc_segs[label] = np.concatenate(( pc_segs[label], pc[i,:][:,np.newaxis].transpose()), axis=0, out=None)
    #
    # idx_keep = []
    # max_dist_local = 4
    # for i in range(len(pc_segs)):
    #    if len(pc_segs[i]) > 200:
    #        idx_keep.append(i)
    #        save_ply('test_' +str(i)+'.ply', pc_segs[i])
    #
    #        import pdb; pdb.set_trace()
    # pc_segs = (np.array(pc_segs)[idx_keep]).tolist()
    #
    #
    #
    # show_pc_segments(pc, pc, pc_segs)

    if len(pc) == 0:
        return [], pc

    # DBSCAN is much faster with metric='precomputed'
    # db1 = dbscan(D, metric='precomputed', eps=0.85, min_samples=2)
    #
    # similarities = euclidean_distances(pc)
    #
    # print(end - start)
    # start = time.time()
    # similarities = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(pc, 'cityblock') )
    # end = time.time()
    # print(end - start)

    # import pdb; pdb.set_trace()
    # start = time.time()
    db = DBSCAN(
        eps=eps,
        metric="euclidean",
        min_samples=min_point_num,
        n_jobs=4,
        leaf_size=leaf_size,
    ).fit(
        pc
    )  # StandardScaler().fit_transform(pc))
    # end = time.time()
    # print('dbscan time = ',end - start)

    num_seg = db.labels_.max() + 1
    pc_segs = [None] * num_seg
    visited = [None] * num_seg
    # import pdb; pdb.set_trace()
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

    # idx_keep = []
    max_dist_local = local_th
    # for i in range(len(pc_segs)):
    # if len(pc_segs[i]) > min_point_num:
    #    idx_keep.append(i)

    # try:

    pc_segs = np.array(pc_segs)  # .tolist()#[idx_keep]).tolist()
    idx_keep = []
    for i in range(len(pc_segs)):
        if len(pc_segs) < 20000:
            if (not is_car(pc_segs[i], min_point_num)) and (not no_filter):
                continue  # output.remove(pc_segs[i])
        idx_keep.append(i)

    if len(pc_segs) > 1:
        pc_segs = pc_segs[idx_keep]
        pc_segs = pc_segs.tolist()
    else:
        tmp = []
        tmp.append(pc_segs)
        pc_segs = tmp
    # import pdb; pdb.set_trace()

    return pc_segs, pc


def pcl_to_numpy(pc):

    cluster_indices = np.arange(pc.size).tolist()
    cloud_clusters = []

    points = np.zeros((len(cluster_indices), 3), dtype=np.float32)
    for i in cluster_indices:
        points[i, 0] = pc[i][0]
        points[i, 1] = pc[i][1]
        points[i, 2] = pc[i][2]

    # for j, indices in enumerate(cluster_indices):
    #    # print('indices = ' + str(len(indices)))
    #    points = np.zeros((len(cluster_indices), 3), dtype=np.float32)
    #
    #    for i in enumerate(cluster_indices):
    #        points[0] = pc[indice][0]
    #        points[1] = pc[indice][1]
    #        points[2] = pc[indice][2]
    #    #print(points.shapepc
    #    cloud_clusters.append(points)

    return points


def pc_segmentation(pc):
    # vg = pc.make_voxel_grid_filter()
    # vg.set_leaf_size(0.01, 0.01, 0.01)
    # cloud_filtered = vg.filter()
    # tree = cloud_filtered.make_kdtree()

    # seg = pc.make_segmenter()
    # seg.set_optimize_coefficients(True)
    # seg.set_model_type(pcl.SACMODEL_PLANE)
    # seg.set_method_type(pcl.SAC_RANSAC)
    # seg.set_distance_threshold(0.001)
    # cluster_indices, coefficients = seg.segment()
    # inds = np.ones((pc.width,1))
    # inds[cluster_indices] = 0
    # return inds

    # segment = pcl.ConditionalEuclideanClustering()
    # cluster_indices = segment.Extract()

    # segment = cloud_filtered.make_RegionGrowing(ksearch=50)
    # segment.set_MinClusterSize(100)
    # segment.set_MaxClusterSize(25000)
    # segment.set_NumberOfNeighbours(5)
    # segment.set_SmoothnessThreshold(0.2)
    # segment.set_CurvatureThreshold(0.05)
    # segment.set_SearchMethod(tree)
    # cluster_indices = segment.Extract()

    cloud_filtered = pc
    segment = cloud_filtered.make_EuclideanClusterExtraction()
    segment.set_ClusterTolerance(0.5)
    segment.set_MinClusterSize(30)
    segment.set_MaxClusterSize(10000)
    # segment.set_SearchMethod(tree)
    cluster_indices = segment.Extract()

    cloud_cluster = pcl.PointCloud()
    # import pdb; pdb.set_trace()
    # print('cluster_indices : ' + str(len(cluster_indices)) + " count.")
    cloud_clusters = []
    for j, indices in enumerate(cluster_indices):
        # print('indices = ' + str(len(indices)))
        points = np.zeros((len(indices), 3), dtype=np.float32)

        for i, indice in enumerate(indices):
            points[i][0] = cloud_filtered[indice][0]
            points[i][1] = cloud_filtered[indice][1]
            points[i][2] = cloud_filtered[indice][2]
        # print(points.shape)
        cloud_clusters.append(points)
        # cloud_clusters[j].from_array(points)
    return cloud_clusters


from mayavi import mlab


def show_pc_segments(pc_all, pc_all2, pc_segments):
    mlab.figure(bgcolor=(0.0, 0.0, 0.0))
    mlab.points3d(
        pc_all[:, 0],
        pc_all[:, 1],
        pc_all[:, 2],
        scale_factor=0.1,
        color=(0.3, 0.2, 0.2),
        opacity=0.3,
    )
    mlab.points3d(
        pc_all2[:, 0],
        pc_all2[:, 1],
        pc_all2[:, 2],
        scale_factor=0.1,
        color=(0.4, 0.4, 0.4),
        opacity=1.0,
    )

    for i in range(len(pc_segments)):
        color = (
            random.random() * 0.8 + 0.2,
            random.random() * 0.8 + 0.2,
            random.random() * 0.8 + 0.2,
        )
        nodes1 = mlab.points3d(
            pc_segments[i][:, 0],
            pc_segments[i][:, 1],
            pc_segments[i][:, 2],
            scale_factor=0.1,
            color=color,
            opacity=0.8,
        )

        nodes1.glyph.scale_mode = "scale_by_vector"

    length_axis = 2.0
    linewidth_axis = 0.1
    color_axis = (1.0, 0.0, 0.0)
    pc_axis = np.array(
        [[0, 0, 0], [length_axis, 0, 0], [0, length_axis, 0], [0, 0, length_axis]]
    )
    mlab.plot3d(
        pc_axis[0:2, 0],
        pc_axis[0:2, 1],
        pc_axis[0:2, 2],
        tube_radius=linewidth_axis,
        color=(1.0, 0.0, 0.0),
    )
    mlab.plot3d(
        pc_axis[[0, 2], 0],
        pc_axis[[0, 2], 1],
        pc_axis[[0, 2], 2],
        tube_radius=linewidth_axis,
        color=(0.0, 1.0, 0.0),
    )
    mlab.plot3d(
        pc_axis[[0, 3], 0],
        pc_axis[[0, 3], 1],
        pc_axis[[0, 3], 2],
        tube_radius=linewidth_axis,
        color=(0.0, 0.0, 1.0),
    )

    mlab.show()


def pc_remove_ground(pc_np, ground_range, plane_fitting=True, ground_removal_th=0.25):

    if plane_fitting:
        view_range = 100  # only process point cloud within 100m
        inds = (
            (pc_np[:, 0] < view_range)
            & (pc_np[:, 1] < view_range)
            & (pc_np[:, 0] > -view_range)
            & (pc_np[:, 1] > -view_range)
        )

        pc_np = pc_np[inds]

        #
        plane_range_th = 100

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

        # import pdb; pdb.set_trace()
        car_height_max = 2.0
        errors = np.matmul(pc_np, normal) + d
        pc_np = pc_np[(errors > ground_removal_th) & (errors < car_height_max), :]
        print("ground normal = ", normal)

    else:
        view_range = 300
        inds = (
            (pc_np[:, 0] < view_range)
            & (pc_np[:, 1] < view_range)
            & (pc_np[:, 2] > ground_range)
            & (pc_np[:, 0] > -view_range)
            & (pc_np[:, 1] > -view_range)
        )

        pc_np = pc_np[inds]

    return pc_np


def get_pc_segments(pc_np):
    view_range = 300
    ground_range = 0.3

    inds = (
        (pc_np[:, 0] < view_range)
        & (pc_np[:, 1] < view_range)
        & (pc_np[:, 2] > ground_range)
        & (pc_np[:, 0] > -view_range)
        & (pc_np[:, 1] > -view_range)
    )

    pc_np = pc_np[inds]

    pc = pcl.PointCloud(pc_np.astype(np.float32))
    pc_segments = pc_segmentation(pc)

    return pc_segments


def get_pc_segments_from_label(pc_raw, path_labels, path_pose):

    with open(path_pose) as f:
        data = json.load(f)
        w_global = data["rotation"]
        t_global = data["translation"]
        R_world = Quaternion(
            w_global["w"], w_global["x"], w_global["y"], w_global["z"]
        ).rotation_matrix
        t_world = np.array([t_global["x"], t_global["y"], t_global["z"]])

    with open(path_labels) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter="\t")
        line_count = 0
        is_header = True
        cuboids = []
        pcs = []
        track_ids = []

        for row in csv_reader:

            if is_header:
                is_header = False
                fieldnames = row
                continue

            if (
                row[13] != "VEHICLE" and row[6] != "False"
            ):  # row[6] != 'True' or row[13] != 'VEHICLE':
                continue

            # direc = path_output + row[2]
            # if not  os.path.exists(direc):
            #    os.makedirs(direc)
            #    with open(direc + '/car_logs.csv', 'w', newline='') as csvfile:
            #        writer = csv.writer(csvfile)
            #        writer.writerow(fieldnames)
            #

            # with open(path_pose, 'a', newline='') as csvfile:
            #    writer = csv.writer(csvfile)
            #    writer.writerow(row)
            track_ids.append(row[2])
            center = {"x": float(row[1]), "y": float(row[5]), "z": float(row[0])}
            rotation = {
                "x": float(row[3]),
                "y": float(row[4]),
                "z": float(row[7]),
                "w": float(row[14]),
            }
            dimensions = {
                "length": float(row[9]),
                "width": float(row[12]),
                "height": float(row[11]),
            }
            cuboids.append(
                {"center": center, "dimensions": dimensions, "rotation": rotation}
            )

            # import pdb; pdb.set_trace()

    pcs, pcs_global, R_world, t_world, cuboids_select, track_ids_select = get_pc_from_cuboid_world_only(
        pc_raw, cuboids, 0, 0, R_world, t_world, track_ids, crop_image=False
    )

    return pcs, pcs_global, track_ids_select, cuboids_select, R_world, t_world


# theta = [-.031, .4, .59]
# rot_x = [[1, 0, 0],
#         [0, cos(theta[0]), -sin(theta[0])],
#         [0, sin(theta[0]), cos(theta[0])]]
# rot_y = [[cos(theta[1]), 0, sin(theta[1])],
#         [0, 1, 0],
#         [-sin(theta[1]), 0, cos(theta[1])]]
# rot_z = [[cos(theta[2]), -sin(theta[1]), 0],
#         [sin(theta[2]), cos(theta[1]), 0],
#         [0, 0, 1]]
# transform = np.dot(rot_x, np.dot(rot_y, rot_z))


def run_icp(data):
    delta_theta_z, delta_x, delta_y, pc_in, pc_out, iter_t, iter_x, iter_y = data
    # if do_exhaustive_serach:
    transf_ini = np.eye(4)
    transf_ini[0, 0] = np.cos(delta_theta_z[iter_t])
    transf_ini[0, 1] = -np.sin(delta_theta_z[iter_t])
    transf_ini[1, 0] = np.sin(delta_theta_z[iter_t])
    transf_ini[1, 1] = np.cos(delta_theta_z[iter_t])
    transf_ini[0, 3] = delta_x[iter_x]  # don't try translation
    transf_ini[1, 3] = delta_y[iter_y]

    pc_in_try = (
        np.matmul(transf_ini[0:3, 0:3], pc_in.transpose())
        + transf_ini[0:3, 3][:, np.newaxis]
    )
    pc_in_try = pc_in_try.transpose()

    cloud_in = pcl.PointCloud()
    cloud_out = pcl.PointCloud()

    # pc_out = np.dot(pc_in, transform)

    cloud_in.from_array(pc_in_try.astype(np.float32))
    cloud_out.from_array(pc_out.astype(np.float32))

    gicp = cloud_in.make_GeneralizedIterativeClosestPoint()

    # icp = cloud_in.make_IterativeClosestPoint()
    converged, transf_iter, estimate, fitness = gicp.gicp(
        cloud_in, cloud_out, max_iter=1000
    )

    transf = np.eye(4)
    transf[0:3, 0:3] = np.matmul(transf_iter[0:3, 0:3], transf_ini[0:3, 0:3])
    transf[0:3, 3] = (
        np.matmul(transf_iter[0:3, 0:3], transf_ini[0:3, 3]) + transf_iter[0:3, 3]
    )

    return transf, fitness
    # transf = np.matmul(transf_iter,transf_ini)


folder_icp_test = "icp_test/"
from multiprocessing.pool import ThreadPool
from multiprocessing import Pool


def get_icp_measurement(
    pc_in_raw, pc_out_raw, center_in, index, do_exhaustive_serach=True
):

    if len(pc_in_raw) <= 20 or len(pc_out_raw) <= 20:
        return np.eye(4), 0, pc_out_raw, 2
    # center_in = np.sum(pc_in,axis=0)/len(pc_in)
    # center_out = np.sum(pc_out,axis=0)/len(pc_out)
    # import pdb; pdb.set_trace()
    pc_in = pc_in_raw - center_in.transpose()
    pc_out = pc_out_raw - center_in.transpose()
    pc_in[:, 2] = 0  # heuristic
    pc_out[:, 2] = 0

    transf_ini = np.eye(4)

    num_try = 1

    if do_exhaustive_serach:
        delta_theta_z = np.arange(-0.15, 0.1501, 0.05) * math.pi
        delta_x = [0]  # np.arange(-0.3, 0.3001, 0.5) #np.arange(-3, 3, 0.5)
        delta_y = [0]  # np.arange(-0.3, 0.3001, 0.5) #np.arange(-3, 3, 0.5)
        # num_try = len(delta_theta_z) * len(delta_t_x) * len(delta_t_y)
    else:
        delta_theta_z = [0]
        delta_x = [0]
        delta_y = [0]

    num_total = len(delta_theta_z) * len(delta_x) * len(delta_y)

    p = Pool(processes=4)
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
                data_input.append(data)

                # transf, fitness = run_icp(delta_theta_z,delta_x,delta_y, pc_in,pc_out, iter_t, iter_x, iter_y)
                # if do_exhaustive_serach:
                # transf_ini = np.eye(4)
                # transf_ini[0,0] =  np.cos(delta_theta_z[iter_t])
                # transf_ini[0,1] = -np.sin(delta_theta_z[iter_t])
                # transf_ini[1,0] =  np.sin(delta_theta_z[iter_t])
                # transf_ini[1,1] =  np.cos(delta_theta_z[iter_t])
                # transf_ini[0,3] =  delta_x[iter_x] #don't try translation
                # transf_ini[1,3] =  delta_y[iter_y]
        #
        #
        # pc_in_try = np.matmul(transf_ini[0:3, 0:3], pc_in.transpose()) + transf_ini[0:3,3][:,np.newaxis]
        # pc_in_try = pc_in_try.transpose()
        #
        # cloud_in = pcl.PointCloud()
        # cloud_out = pcl.PointCloud()
        #
        # cloud_in.from_array(pc_in_try.astype(np.float32))
        # cloud_out.from_array(pc_out.astype(np.float32))
        #
        # gicp = cloud_in.make_GeneralizedIterativeClosestPoint()
        # converged, transf_iter, estimate, fitness = gicp.gicp(cloud_in, cloud_out,max_iter=1000)
    #
    #
    # transf = np.eye(4)
    # transf[0:3,0:3] = np.matmul(transf_iter[0:3,0:3],transf_ini[0:3,0:3])
    # transf[0:3,3] =  np.matmul(transf_iter[0:3,0:3],transf_ini[0:3,3]) +  transf_iter[0:3,3]

    # idx = iter_t * (len(delta_x) * len(delta_y)) + iter_x * len(delta_x) + iter_y #* len(delta_y)
    # transfs[idx] = transf
    # fitnesss[idx] = fitness
    #

    # cloud_out = np.dot(cloud_in , transf)
    # print('has converged:' + str(converged) + ' score: ' + str(fitness) )
    # print(str(transf))
    # save_ply(folder_icp_test + str(iter_t)+ str(iter_x)+ str(iter_y)  + '_pc_in_try'  + '.ply',pc_in_try)
    # save_ply(folder_icp_test + str(iter_t)+ str(iter_x)+ str(iter_y)  + '_pc_out_try' + '.ply',pc_out)
    # save_ply(folder_icp_test + str(iter_t)+ str(iter_x)+ str(iter_y)  + '_pc_out2_try'+ '.ply', np.dot(pc_in,transf[0:3,0:3].transpose()) + transf[0:3,3])

    results = p.map(run_icp, data_input)
    p.close()

    transfs = [None] * num_total
    fitness = [None] * num_total
    for i in range(num_total):
        transfs[i] = results[i][0]
        fitness[i] = results[i][1]

    # np.matmul(transf[0:3,0:3], pc_in.transpose()) + transf[0:3,3]

    # transf[0:3,3] = -np.matmul(transf[0:3,0:3],center_in)+transf[0:3,3]+center_out
    fitness = np.array(fitness)
    transf_best = transfs[np.argmin(fitness)]
    # import pdb; pdb.set_trace()

    # print('*****debug**')
    # transf_best = check_transf(transf_best_out)
    # print(transf_best_out)
    is_good = check_transf(transf_best)
    # print(transf_best)
    # print(is_good)

    transf_best[0:3, 3] += np.matmul(np.eye(3) - transf_best[0:3, 0:3], center_in)[:, 0]
    # import pdb; pdb.set_trace()

    pc_aligned = (
        np.dot(pc_in_raw, transf_best[0:3, 0:3].transpose()) + transf_best[0:3, 3]
    )
    pc_accu = np.concatenate((pc_aligned, pc_out_raw), axis=0)
    # save_ply(folder_icp_test + str(index)+'_pc_in.ply',pc_in_raw)
    # save_ply(folder_icp_test + str(index)+'_pc_aligned.ply',pc_aligned)
    # save_ply(folder_icp_test + str(index)+'_pc_out.ply',pc_out_raw)
    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()
    return transf_best, fitness.min(), pc_accu, is_good


def get_icp_measurement_for_wrold(pc_in_raw, pc_out_raw, center_in):

    # center_in = np.sum(pc_in,axis=0)/len(pc_in)
    pc_in = pc_in_raw - center_in.transpose()
    pc_out = pc_out_raw - center_in.transpose()
    pc_in[:, 2] = 0  # heuristic
    pc_out[:, 2] = 0

    transf_ini = np.eye(4)

    num_try = 1

    # if do_exhaustive_serach:
    #    delta_theta_z = np.arange(-0.2,0.2001, 0.1)*math.pi
    #    delta_x = np.arange(-0.5, 0.5001, 0.25) #np.arange(-3, 3, 0.5)
    #    delta_y = np.arange(-0.5, 0.5001, 0.25) #np.arange(-3, 3, 0.5)
    #    #num_try = len(delta_theta_z) * len(delta_t_x) * len(delta_t_y)
    # else:
    delta_theta_z = [0]
    delta_x = [0]
    delta_y = [0]

    transfs = []
    fitnesss = []
    for iter_t in range(len(delta_theta_z)):
        for iter_x in range(len(delta_x)):
            for iter_y in range(len(delta_y)):

                # if do_exhaustive_serach:
                transf_ini = np.eye(4)
                transf_ini[0, 0] = np.cos(delta_theta_z[iter_t])
                transf_ini[0, 1] = -np.sin(delta_theta_z[iter_t])
                transf_ini[1, 0] = np.sin(delta_theta_z[iter_t])
                transf_ini[1, 1] = np.cos(delta_theta_z[iter_t])
                transf_ini[0, 3] = delta_x[iter_x]  # don't try translation
                transf_ini[1, 3] = delta_y[iter_y]

                pc_in_try = (
                    np.matmul(transf_ini[0:3, 0:3], pc_in.transpose())
                    + transf_ini[0:3, 3][:, np.newaxis]
                )
                pc_in_try = pc_in_try.transpose()

                cloud_in = pcl.PointCloud()
                cloud_out = pcl.PointCloud()

                # pc_out = np.dot(pc_in, transform)

                cloud_in.from_array(pc_in_try.astype(np.float32))
                cloud_out.from_array(pc_out.astype(np.float32))

                gicp = cloud_in.make_GeneralizedIterativeClosestPoint()

                # icp = cloud_in.make_IterativeClosestPoint()
                converged, transf_iter, estimate, fitness = gicp.gicp(
                    cloud_in, cloud_out, max_iter=1000
                )

                transf = np.eye(4)
                transf[0:3, 0:3] = np.matmul(
                    transf_iter[0:3, 0:3], transf_ini[0:3, 0:3]
                )
                transf[0:3, 3] = (
                    np.matmul(transf_iter[0:3, 0:3], transf_ini[0:3, 3])
                    + transf_iter[0:3, 3]
                )

                # transf = np.matmul(transf_iter,transf_ini)

                transfs.append(transf)
                fitnesss.append(fitness)
                # cloud_out = np.dot(cloud_in , transf)
                # print('has converged:' + str(converged) + ' score: ' + str(fitness) )
                # print(str(transf))
                # save_ply(folder_icp_test + str(iter_t)+ str(iter_x)+ str(iter_y)  + '_pc_in_try'  + '.ply',pc_in_try)
                # save_ply(folder_icp_test + str(iter_t)+ str(iter_x)+ str(iter_y)  + '_pc_out_try' + '.ply',pc_out)
                # save_ply(folder_icp_test + str(iter_t)+ str(iter_x)+ str(iter_y)  + '_pc_out2_try'+ '.ply', np.dot(pc_in,transf[0:3,0:3].transpose()) + transf[0:3,3])

    # np.matmul(transf[0:3,0:3], pc_in.transpose()) + transf[0:3,3]

    # transf[0:3,3] = -np.matmul(transf[0:3,0:3],center_in)+transf[0:3,3]+center_out
    fitnesss = np.array(fitnesss)
    transf_best = transfs[np.argmin(fitnesss)]
    # import pdb; pdb.set_trace()
    transf_best[0:3, 3] += np.matmul(np.eye(3) - transf_best[0:3, 0:3], center_in)[:, 0]
    # import pdb; pdb.set_trace()

    pc_aligned = (
        np.dot(pc_in_raw, transf_best[0:3, 0:3].transpose()) + transf_best[0:3, 3]
    )

    # save_ply(folder_icp_test + str(index)+'_pc_in.ply',pc_in_raw)
    # save_ply(folder_icp_test + str(index)+'_pc_aligned.ply',pc_aligned)
    # save_ply(folder_icp_test + str(index)+'_pc_out.ply',pc_out_raw)
    return transf_best, fitnesss.min()


USE_IN_ACCU_AND_EKF = 0
USE_IN_ACCU_BUT_EKF = 1
USE_IN_NOT_ACCU_OR_EKF = 2


def check_transf(transf):
    dx = transf[0, 3]
    dy = transf[1, 3]
    theta_z = np.arctan2(transf[1, 0], transf[0, 0])

    theta_th = np.sqrt(dx ** 2 + dy ** 2) / CAR_TURNING_RADIUS
    # print(theta_z, theta_th)
    # radius = 10 #Motor Trend refers to a curb-to-curb turning circle of a 2008 Cadillac CTS as 35.5 feet (10.82 m)

    if (theta_z > -theta_th) and (theta_z < theta_th):
        return USE_IN_ACCU_AND_EKF

    if np.sqrt(dx ** 2 + dy ** 2) < 0.1 and abs(theta_z) < 0.05:
        return USE_IN_ACCU_BUT_EKF

    return USE_IN_NOT_ACCU_OR_EKF

    # if theta_z > 0:
    #    theta_z = min(theta_z, theta_th)
    # else:
    #    theta_z = max(theta_z, -theta_th)


#
#
#
# transf_new = transf.copy()
# transf_new[2,3]=0
# transf_new[2,2]=1
# transf_new[0,0:2] = [np.cos(theta_z), -np.sin(theta_z)]
# transf_new[1,0:2] = [np.sin(theta_z), np.cos(theta_z)]
# transf_new[0,2] = 0
# transf_new[1,2] = 0
# transf_new[2,0] = 0
# transf_new[2,1] = 0
#
#
# return transf_new


def tranf_to_z(
    x_prev, transf, center_in
):  # convert transformation matrix to measurement

    # theta_z = np.arccos( (transf[0,0] + transf[1,1] + transf[2,2]-1)/2)
    theta_z = np.arctan2(transf[1, 0], transf[0, 0])

    # import pdb; pdb.set_trace()
    # t_pose = transf[0:3,3] - np.matmul(np.eye(3) - transf[0:3,0:3], x_prev)[:,0]
    # dx = t_pose[0]
    # dy = t_pose[1]

    pose_old = x_prev.copy()
    pose_old[2, 0] = 0
    pose_new = np.matmul(transf[0:3, 0:3], pose_old) + transf[0:3, 3][:, np.newaxis]

    dx = pose_new[0, 0] - x_prev[0, 0]
    dy = pose_new[1, 0] - x_prev[1, 0]
    # print('in tranf_to_z : ')
    # print(theta_z)
    if True:
        theta_th = np.sqrt(dx ** 2 + dy ** 2) / CAR_TURNING_RADIUS
        # radius = 10 #Motor Trend refers to a curb-to-curb turning circle of a 2008 Cadillac CTS as 35.5 feet (10.82 m)
        if theta_z > 0:
            theta_z = min(theta_z, theta_th)
        else:
            theta_z = max(theta_z, -theta_th)

    # print(theta_th)
    # angle = acos(( m00 + m11 + m22 - 1)/2)
    # x = (m21 - m12)/√((m21 - m12)2+(m02 - m20)2+(m10 - m01)2)
    # y = (m02 - m20)/√((m21 - m12)2+(m02 - m20)2+(m10 - m01)2)
    # z = (m10 - m01)/√((m21 - m12)2+(m02 - m20)2+(m10 - m01)2)
    output = x_prev + np.array([0, 0, theta_z])[:, np.newaxis]
    output[0, 0] = pose_new[0, 0]
    output[1, 0] = pose_new[1, 0]
    # print(output)

    # print(output)
    # limit angular velocity
    # import pdb; pdb.set_trace()

    # if np.linalg.norm(x_prev[0:2,0])

    return output


def vector_proj(v_input, v_target):
    v_target /= np.linalg.norm(v_target[0:2, 0])
    return np.dot(v_input[:, 0], v_target[:, 0]) * v_target


def compute_theta_diff(theta1, theta2):

    diff = theta1 - theta2

    if diff > 2 * math.pi:
        diff -= 2 * math.pi
    elif diff < -2 * math.pi:
        diff += 2 * math.pi

    if abs(diff) > math.pi:
        return 2 * math.pi - abs(diff)

    else:
        return abs(diff)


def compute_theta_complement(theta):
    theta = math.pi + theta
    if theta > math.pi:
        return theta - 2 * math.pi
    else:
        return theta


def do_EKF(
    count,
    x_prev,
    Sigma,
    z,
    u,
    center_seg,
    pc_segment,
    icp_score,
    icp_ratio=1,
    motion_model="static",
):
    # follow book notation

    # use point segment center as z...(instead of ICP)
    # center_pc = pc_segment.mean(axis=0)
    # import pdb; pdb.set_trace()
    z[0:2, 0] = center_seg[0:2]

    icp_score = max(0, icp_score)

    R = np.diag([0.1, 0.1, math.pi / 18])  # *1000000
    Q = np.diag([0.1, 0.1, math.pi / 1000]) * min(np.exp(icp_score * 50), 1000)  # /2

    # trust ICP for initial velocity estimation
    if icp_ratio == 0:
        Q[2, 2] *= 10000000
    # else:
    #    if count < 3:
    #        R *= 10000000
    #    Q /= 10

    # if count < 5:
    # Q /= icp_ratio

    A = np.eye(3)
    C = np.eye(3)
    B = np.eye(3)

    #
    # use heuristics for motion model and covariance matrix
    # if theta difference is too large, increase variace and clamp it

    # dist_th = 1.5
    # it can never turn unless ICP ask it to turn...
    theta_th = (
        np.sqrt(u[0, 0] ** 2 + u[1, 0] ** 2) / CAR_TURNING_RADIUS
    )  # math.pi/18 #cannot rotate much once
    theta_diff = np.abs(z[2, 0] - x_prev[2, 0])
    if True:
        Q = Q * np.exp(2 * theta_th)

        # if theta_diff >0:
        #     Q = Q *np.exp(max(0, -theta_th))
        # else:
        #     Q = Q *np.exp(max(0, np.abs(z[2,0]+theta_th)))

    # heta_diff = np.abs(z[2,0] - x_prev[2,0])
    # if theta_diff > theta_th :
    #    Q[2,2] *= max(np.exp(max(0, np.abs(z[2,0])-theta_th)),1)
    # import pdb; pdb.set_trace()

    theta_from_u = np.arctan2(u[1, 0], u[0, 0])
    theta_diff_from_u = theta_from_u - x_prev[2, 0]

    # if np.linalg.norm(u[0:2,0]) > dist_th:
    # u /=np.linalg.norm(u)

    if theta_diff_from_u > 0:
        theta_pred = min(theta_diff_from_u, theta_th) + x_prev[2, 0]
    else:
        theta_pred = max(theta_diff_from_u, -theta_th) + x_prev[2, 0]
        #

        # R[2,2] /= np.exp(max(0,np.linalg.norm(u[0:2,0])-dist_th))
        # import pdb; pdb.set_trace()

    # else:
    #    theta_pred = x_prev[2,0] + u[2,0]

    # cars can only move forward
    # import pdb; pdb.set_trace()

    # R_prev = np.array([[np.cos(x_prev[2,0]), -np.sin(x_prev[2,0])],[np.sin(x_prev[2,0]), np.cos(x_prev[2,0])]])
    # axis_prev = np.matmul(R_prev, axis0)
    # v_pred_proj = np.dot(v_pred[:,0], axis_prev[:,0]) * axis_prev

    # limit the range of velocity vector

    v_pred_prev = np.array(
        [center_seg[0] - x_prev[0, 0], center_seg[1] - x_prev[1, 0]]
    )[:, np.newaxis]
    if np.linalg.norm(v_pred_prev) > 1.5:  # motion_model == 'const_v':
        ratio = 0.3

        axis0 = np.array([1, 0])[:, np.newaxis]
        R_prev = np.array(
            [
                [np.cos(x_prev[2, 0]), -np.sin(x_prev[2, 0])],
                [np.sin(x_prev[2, 0]), np.cos(x_prev[2, 0])],
            ]
        )
        axis_prev = np.matmul(R_prev, axis0)

        R1 = np.array(
            [
                [np.cos(theta_th), -np.sin(theta_th)],
                [np.sin(theta_th), np.cos(theta_th)],
            ]
        )
        v1 = np.matmul(R1, axis_prev)
        v2 = np.matmul(R1.transpose(), axis_prev)

        if (np.cross(v_pred_prev[:, 0], v1[:, 0])) * (
            np.cross(v_pred_prev[:, 0], v2[:, 0])
        ) > 0:
            if np.cross(v_pred_prev[:, 0], v1[:, 0]) > 0:
                # v_pred_proj = np.dot(v_pred_prev[:,0], axis_prev[:,0]) * axis_prev
                v_pred = vector_proj(v_pred_prev, v1)
            else:
                v_pred = vector_proj(v_pred_prev, v2)
        else:
            v_pred = v_pred_prev

        # import pdb; pdb.set_trace()

        x_pred = np.array(
            [v_pred[0, 0] + x_prev[0, 0], v_pred[1, 0] + x_prev[1, 0], theta_pred]
        )[
            :, np.newaxis
        ]  # np.matmul(A,x_prev) + np.matmul(B,u)
    else:
        # elif motion_model == 'static':
        x_pred = x_prev
    # else:
    #    print('Wrong motion model!')

    # if np.linalg.norm(x_pred - center_seg) < 1.0:
    ratio2 = 0.5
    v_theta = np.array([np.cos(x_prev[2, 0]), np.sin(x_prev[2, 0])])
    x_diff = np.array([(center_seg[0] - x_pred[0, 0]), (center_seg[1] - x_pred[1, 0])])
    x_diff_proj = np.dot(v_theta, x_diff) * v_theta

    # import pdb;pdb.set_trace()
    x_pred[0, 0] += x_diff_proj[0] * ratio2 + (x_diff - x_diff_proj)[0] * 0.1
    x_pred[1, 0] += x_diff_proj[1] * ratio2 + (x_diff - x_diff_proj)[1] * 0.1

    # import pdb; pdb.set_trace()
    theta_from_u = np.arctan2(u[1, 0], u[0, 0])

    bbox_min = MinimumBoundingBox.MinimumBoundingBox(pc_segment[:, 0:2])
    l01 = bbox_min.length_parallel
    l02 = bbox_min.length_orthogonal

    if l01 / l02 > 1.5:
        theta_from_bbox = bbox_min.unit_vector_angle
        if compute_theta_diff(theta_from_bbox, x_pred[2, 0]) > math.pi / 2:
            theta_from_bbox = compute_theta_complement(theta_from_bbox)

        theta_est = theta_from_bbox

    elif np.linalg.norm(u) > 1:
        theta_est = theta_from_u
    else:
        theta_est = x_pred[2, 0]

    ratio = 0.5
    x_pred[2, 0] = theta_est * (1 - ratio) + x_pred[2, 0] * ratio

    # print('x_pred = ', x_pred)
    # print('z = ', z)

    # x_pred = np.array([center_seg[0], center_seg[1], x_prev[2,0] + u[2,0]])[:, np.newaxis] #np.matmul(A,x_prev) + np.matmul(B,u)

    Sigma_pred = np.matmul(np.matmul(A, Sigma), A.transpose()) + R

    K = np.matmul(
        np.matmul(Sigma_pred, C.transpose()),
        np.linalg.inv(np.matmul(C, np.matmul(Sigma_pred, C.transpose())) + Q),
    )

    x_est = x_pred + np.matmul(K, z - np.matmul(C, x_pred))

    Sigma_est = np.matmul((np.eye(3) - np.matmul(K, C)), Sigma_pred)
    # import pdb; pdb.set_trace()

    if x_est[2, 0] > math.pi:
        x_est[2, 0] = 2 * math.pi - x_est[2, 0]

    elif x_est[2, 0] < -math.pi:
        x_est[2, 0] = 2 * math.pi + x_est[2, 0]

    return x_est, Sigma_est


def transform_bbox(bbox, x):
    R = np.array(
        [
            [np.cos(x[2, 0]), -np.sin(x[2, 0]), 0],
            [np.sin(x[2, 0]), np.cos(x[2, 0]), 0],
            [0, 0, 1],
        ]
    )
    t = np.array([x[0, 0], x[1, 0], 0])[:, np.newaxis]
    # import pdb; pdb.set_trace()
    return transform_bounding_box_3d(bbox, R, t)


# def get_initial_x_from_bbox(bbox):
#
#    x = compute_bbox_center(bbox)
#
#    axis0 = np.array([1,0,0])
#    import pdb; pdb.set_trace()
#    axis_now = bbox[1]-bbox[0]
#
#    x[3] =  atan2d(axis0[0]*axis_now[1]-axis_now[0]*axis0[1],axis0[0]*axis0[1]+axis_now[0]*axis_now[1])
#
#    return x


def get_bbox_from_label(bbox_label, R_world, t_world):
    bbox_raw = BBox3D(
        bbox_label["center"]["x"],
        bbox_label["center"]["y"],
        bbox_label["center"]["z"],
        bbox_label["dimensions"]["length"],
        bbox_label["dimensions"]["width"],
        bbox_label["dimensions"]["height"],
        rx=bbox_label["rotation"]["x"],
        ry=bbox_label["rotation"]["y"],
        rz=bbox_label["rotation"]["z"],
        rw=bbox_label["rotation"]["w"],
    )
    h = np.linalg.norm(bbox_raw.p5[0:3] - bbox_raw.p1[0:3])
    bbox = [
        bbox_raw.p1[0:3][:, np.newaxis],
        bbox_raw.p2[0:3][:, np.newaxis],
        bbox_raw.p4[0:3][:, np.newaxis],
        h,
    ]

    bbox = transform_bounding_box_3d(bbox, R_world, t_world)

    return bbox


def get_bbox0_from_label(bbox_label, R_world, t_world):  # align bbox with x axis
    bbox_raw = BBox3D(
        bbox_label["center"]["x"],
        bbox_label["center"]["y"],
        bbox_label["center"]["z"],
        bbox_label["dimensions"]["length"],
        bbox_label["dimensions"]["width"],
        bbox_label["dimensions"]["height"],
        rx=bbox_label["rotation"]["x"],
        ry=bbox_label["rotation"]["y"],
        rz=bbox_label["rotation"]["z"],
        rw=bbox_label["rotation"]["w"],
    )
    h = np.linalg.norm(bbox_raw.p5[0:3] - bbox_raw.p1[0:3])
    bbox = [
        bbox_raw.p1[0:3][:, np.newaxis],
        bbox_raw.p2[0:3][:, np.newaxis],
        bbox_raw.p4[0:3][:, np.newaxis],
        h,
    ]

    bbox = transform_bounding_box_3d(bbox, R_world, t_world)

    x = compute_bbox_center(bbox)
    bbox[0] = bbox[0] - x
    bbox[1] = bbox[1] - x
    bbox[2] = bbox[2] - x

    axis0 = np.array([1, 0, 0])
    axis_now = np.array(bbox[1] - bbox[0])
    axis_now = axis_now / np.linalg.norm(axis_now)

    theta_z = -np.arctan2(axis_now[1], axis_now[0])
    # heta_z =  np.arctan2(axis_now[0]*axis0[1]-axis0[0]*axis_now[1],axis_now[0]*axis_now[1]+axis0[0]*axis0[1])

    # heta_z =  np.arctan2(axis0[0]*axis_now[1]-axis_now[0]*axis0[1],axis0[0]*axis0[1]+axis_now[0]*axis_now[1])
    theta_z = theta_z[0]

    R = np.eye(3)
    R[0, 0] = np.cos(theta_z)
    R[0, 1] = -np.sin(theta_z)
    R[1, 0] = np.sin(theta_z)
    R[1, 1] = np.cos(theta_z)
    #    R = np.array([[np.cos(theta_z), -np.sin(theta_z), 0 ],[, np.cos(theta_z), 0],[0,0,1]])

    t = np.zeros((3, 1))
    # import pdb; pdb.set_trace()
    x_initial = np.array([x[0][0], x[1][0], -theta_z])[:, np.newaxis]
    return transform_bounding_box_3d(bbox, R, t), x_initial


def get_pc_segments_from_label_kitti(
    path_frame, pc_raw, min_point_num, dataset_name="argo"
):

    if dataset_name == "argo":
        R_world = np.load(path_frame + "_city_R_egovehicle.npy")
        t_world = np.load(path_frame + "_city_t_egovehicle.npy")
    else:
        R_world = np.load(path_frame + "_R_world.npy")
        t_world = np.load(path_frame + "_t_world.npy")

    track_id_all = []
    bbox_all = []
    pc_seg_all = []
    pc_seg_world_all = []
    path_bboxs = glob.glob(path_frame + "*_bbox.npy")

    for i in range(len(path_bboxs)):
        path_bbox = path_bboxs[i]
        track_id = path_bbox.split("/")[-1].split("_")[-2]
        bbox = np.load(path_bbox)

        U = []
        V = []
        W = []
        P1 = []
        P2 = []
        P4 = []
        P5 = []
        index = []
        # for c in range(len(cuboids)):
        #    if not 'center' in cuboids[c]:
        #        continue
        # bbox = BBox3D(cuboids[c]['center']['x'], \
        #    cuboids[c]['center']['y'], \
        #    cuboids[c]['center']['z'], \
        #    cuboids[c]['dimensions']['length'], \
        #    cuboids[c]['dimensions']['width'], \
        #    cuboids[c]['dimensions']['height'], \
        #    rx = cuboids[c]['rotation']['x'], \
        #    ry = cuboids[c]['rotation']['y'], \
        #    rz = cuboids[c]['rotation']['z'], \
        #    rw = cuboids[c]['rotation']['w'])

        # u = bbox.p2 - bbox.p1
        # v = bbox.p4 - bbox.p1
        # w = bbox.p5 - bbox.p1
        # import pdb;pdb.set_trace()

        u = bbox[1] - bbox[0]
        v = bbox[2] - bbox[0]
        w = np.zeros((3, 1))  # bbox[0].copy()
        w[2, 0] += bbox[3]

        p5 = w + bbox[0]

        U.append(u[0:3, 0])
        V.append(v[0:3, 0])
        W.append(w[0:3, 0])
        P1.append(bbox[0][0:3, 0])
        P2.append(bbox[1][0:3, 0])
        P4.append(bbox[2][0:3, 0])
        P5.append(p5[0:3, 0])
        # index.append(c)

        if len(U) == 0:
            return (
                pc_seg_world_all,
                pc_seg_all,
                track_id_all,
                bbox_all,
                R_world,
                t_world,
            )

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

        # import pdb; pdb.set_trace()

        pc_seg = pc_raw[flag[0, :]]

        pc_seg_all.append(pc_seg)
        pc_seg_world_all.append(
            (np.matmul(R_world, pc_seg.transpose()) + t_world).transpose()
        )
        bbox_all.append(bbox)
        track_id_all.append(track_id)

    # use all ground truth bboxes
    idx_keep = []
    for i in range(len(pc_seg_world_all)):
        if (
            len(pc_seg_world_all[i]) >= min_point_num
        ):  # and (get_polygon_area(pc_seg_world_all[i]) < 12):
            idx_keep.append(i)

    #    import pdb; pdb.set_trace()

    pc_seg_world_all = np.array(pc_seg_world_all)[idx_keep].tolist()
    pc_seg_all = np.array(pc_seg_all)[idx_keep].tolist()
    track_id_all = np.array(track_id_all)[idx_keep].tolist()
    bbox_all = np.array(bbox_all)[idx_keep].tolist()

    return pc_seg_world_all, pc_seg_all, track_id_all, bbox_all, R_world, t_world


def get_bbox0(bbox, R_world, t_world):  # align bbox with x axis
    # bbox_raw = BBox3D(bbox_label['center']['x'], \
    #        bbox_label['center']['y'], \
    #        bbox_label['center']['z'], \
    #        bbox_label['dimensions']['length'], \
    #        bbox_label['dimensions']['width'], \
    #        bbox_label['dimensions']['height'], \
    #        rx = bbox_label['rotation']['x'], \
    #        ry = bbox_label['rotation']['y'], \
    #        rz = bbox_label['rotation']['z'], \
    #        rw = bbox_label['rotation']['w'])
    # h = np.linalg.norm(bbox_raw.p5[0:3] - bbox_raw.p1[0:3])
    # bbox = [bbox_raw.p1[0:3][:,np.newaxis], bbox_raw.p2[0:3][:, np.newaxis], bbox_raw.p4[0:3][:, np.newaxis], h]

    bbox = transform_bounding_box_3d(bbox, R_world, t_world)

    x = compute_bbox_center(bbox)
    bbox[0] = bbox[0] - x
    bbox[1] = bbox[1] - x
    bbox[2] = bbox[2] - x

    axis0 = np.array([1, 0, 0])
    axis_now = np.array(bbox[1] - bbox[0])
    axis_now = axis_now / np.linalg.norm(axis_now)

    theta_z = -np.arctan2(axis_now[1], axis_now[0])
    # heta_z =  np.arctan2(axis_now[0]*axis0[1]-axis0[0]*axis_now[1],axis_now[0]*axis_now[1]+axis0[0]*axis0[1])

    # heta_z =  np.arctan2(axis0[0]*axis_now[1]-axis_now[0]*axis0[1],axis0[0]*axis0[1]+axis_now[0]*axis_now[1])
    theta_z = theta_z[0]

    R = np.eye(3)
    R[0, 0] = np.cos(theta_z)
    R[0, 1] = -np.sin(theta_z)
    R[1, 0] = np.sin(theta_z)
    R[1, 1] = np.cos(theta_z)
    #    R = np.array([[np.cos(theta_z), -np.sin(theta_z), 0 ],[, np.cos(theta_z), 0],[0,0,1]])

    t = np.zeros((3, 1))
    # import pdb; pdb.set_trace()
    x_initial = np.array([x[0][0], x[1][0], -theta_z])[:, np.newaxis]
    return transform_bounding_box_3d(bbox, R, t), x_initial


import shapely.geometry
import shapely.affinity
from shapely.geometry import Point, Polygon, asPolygon

idx_bbox = np.array([0, 1, 4, 2])
idx_bbox = idx_bbox.astype("int")


def bbox_to_contour(bbox):
    # point_bbox = recover_bounding_box_3d(bbox)
    # c = shapely.geometry.box(bbox)
    #
    # return Polygon(np.concatenate((bbox[idx_bbox,:],bbox[idx_bbox[0],:][:,np.newaxis].transpose()),axis=0))
    return Polygon(bbox[idx_bbox, 0:2])


def get_iou(b1, b2):

    # import pdb; pdb.set_trace()
    b1_c = bbox_to_contour(b1)
    b2_c = bbox_to_contour(b2)

    inter_area = b1_c.intersection(b2_c).area
    union_area = b1_c.area + b2_c.area - inter_area

    return inter_area / union_area


import smallestenclosingcircle


def get_polygon_center(pc):
    # hull = ConvexHull(pc)
    # import pdb; pdb.set_trace()
    # try:
    #    pc_new = pc[hull.vertices,:]
    # except:
    #    import pdb; pdb.set_trace()
    # return np.sum(pc_new, axis = 0)/ len(pc_new)
    # try:

    sample_size = 100
    if len(pc) > sample_size:
        random.sample(np.arange(len(pc)).tolist(), sample_size)

    pc = np.array(pc)
    center = np.sum(pc, axis=0) / len(pc)
    circle = smallestenclosingcircle.make_circle(pc[:, 0:2])
    #    except:
    #        import pdb; pdb.set_trace()
    return np.array([circle[0], circle[1], center[2]])


from scipy.spatial import ConvexHull


def get_polygon_area(pc):

    #    import pdb; pdb.set_trace()
    try:
        pc = np.array(pc)
        hull = ConvexHull(pc[:, 0:2])
        return Polygon(pc[hull.vertices, 0:2]).area
    except:
        return 0


sys.path.append("../../utils")
from utils.se3 import SE3
import copy
from map_representation.map_api import ArgoverseMap

avm = ArgoverseMap()


def leave_only_drivable_region(lidar_pts, R_world, t_world, city_name="MIA"):

    city_to_egovehicle_se3 = SE3(rotation=R_world, translation=t_world[:, 0])
    drivable_area_pts = copy.deepcopy(lidar_pts)
    drivable_area_pts = city_to_egovehicle_se3.transform_point_cloud(
        drivable_area_pts
    )  # put into city coords
    # road_lidar_pts = avm.decimate_point_cloud_to_lane_area(road_lidar_pts, city_name)
    # road_lidar_pts = avm.remove_ground_surface(road_lidar_pts, city_name)
    drivable_area_pts = avm.remove_non_drivable_area_points(
        drivable_area_pts, city_name
    )
    drivable_area_pts = city_to_egovehicle_se3.inverse_transform_point_cloud(
        drivable_area_pts
    )  # put back into ego-vehicle coords
    return drivable_area_pts
    # except:
    #    print("ERROR in remove_non_drivable_area_points!!")
    #    return lidar_pts


# early stopping to accelrate here
def is_car(pc, min_point_num):

    pc = np.array(pc)
    # import pdb; pdb.set_trace()
    # area = get_polygon_area(pc)

    if len(pc) < min_point_num:
        return False

    height = pc[:, 2].max() - pc[:, 2].min()
    if height < 0.5:
        return False

    num_lower = len(pc[pc[:, 2] < (height / 2 + pc[:, 2].min()), :])
    if num_lower / len(pc) > 0.8:
        return False

    bbox_min = MinimumBoundingBox.MinimumBoundingBox(pc[:, 0:2])
    l01 = bbox_min.length_parallel
    l02 = bbox_min.length_orthogonal
    area = l01 * l02

    if (area < 2 or area > 20) or ((l01 > 5 or l02 > 5)) or ((l01 < 0.4 and l02 < 0.4)):
        return False

    # center_seg =  get_polygon_center(pc)
    # dists_local = pc - center_seg
    # dists_local = np.sqrt(np.sum(dists_local**2,axis=1))

    # if  (max(dists_local) > max_dist_local)
    #    return False

    # return True
    # import pdb; pdb.set_trace()

    is_p, tmp, tmp = is_plane(pc, 0.02)

    return not is_p


def initialize_bbox(pc, R_world, t_world, city_name, use_map_lane, fix_bbox_size):

    return smallest_bbox(
        pc, R_world, t_world, city_name, use_map_lane, fix_size=fix_bbox_size
    )

    # bbox = smallest_bbox(pc)


#
# center_seg =  get_polygon_center(pc)c
#
# p0 = np.array([-1,-1,0]) + center_seg
# p1 = np.array([ 1,-1,0]) + center_seg
# p2 = np.array([-1, 1,0]) + center_seg
# h = 1
#
#
# return [p0[:,np.newaxis],p1[:,np.newaxis],p2[:,np.newaxis],h]

# return smallest_bbox(pc)


def get_bbox0():
    p0 = np.array([-1, -1, 0])
    p1 = np.array([1, -1, 0])
    p2 = np.array([-1, 1, 0])
    h = 1
    return [p0[:, np.newaxis], p1[:, np.newaxis], p2[:, np.newaxis], h]


from MinimumBoundingBox import MinimumBoundingBox


def smallest_bbox(
    pc, R_world, t_world, city_name, use_map_lane, fix_size=True, x_initial=[]
):
    # hull = ConvexHull(pc[0:2,:])
    # import pdb; pdb.set_trace()
    if len(pc) > 2:

        bbox_min = MinimumBoundingBox.MinimumBoundingBox(pc[:, 0:2])
        #

        # bbox.area  # 16
        # bbox.rectangle_center  # (1.3411764705882352, 1.0647058823529414)
        # bbox.corner_points

        l01_smallest = bbox_min.length_parallel
        l02_smallest = bbox_min.length_orthogonal
        center = bbox_min.rectangle_center
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

        p_back = center - vec_parllel * l01_smallest / 2
        p0 = p_back + vec_vertical * l02_smallest / 2
        p1 = p0 + vec_parllel * l01_smallest
        p2 = p_back - vec_vertical * l02_smallest / 2

        h_bottom = pc[:, 2].min() - 0.3
        h = pc[:, 2].max() - h_bottom
        p0 = np.concatenate((p0, [h_bottom]), axis=0)
        p1 = np.concatenate((p1, [h_bottom]), axis=0)
        p2 = np.concatenate((p2, [h_bottom]), axis=0)

        bbox = [p0[:, np.newaxis], p1[:, np.newaxis], p2[:, np.newaxis], h]
    else:
        import pdb

        pdb.set_trace()
        l01 = 4.8
        l02 = 2.4
        center = pc.sum(axis=0) / len(pc)
        angle = 0

    if len(x_initial) == 0:
        x_initial = np.array([center[0], center[1], angle])[:, np.newaxis]

    return bbox, x_initial

    if angle > math.pi:
        angle = angle - 2 * math.pi

    # if l01_smallest < l02_smallest:
    #    angle = angle + math.pi/2
    #    if angle > math.pi:
    #        angle = 2*math.pi - angle
    #
    #    tmp = l01_smallest
    #    l01_smallest = l02_smallest
    #    l02_smallest = tmp
    #

    if fix_size:
        l01 = 4.8
        l02 = 2.4
    else:
        l01 = l01_smallest
        l02 = l02_smallest

    # don't transform! use smallest bbox directly....

    h_bottom = pc[:, 2].min() - 0.3
    h = pc[:, 2].max() - h_bottom
    p0 = np.array([-l01 / 2, -l02 / 2, h_bottom])
    p1 = np.array([l01 / 2, -l02 / 2, h_bottom])
    p2 = np.array([-l01 / 2, l02 / 2, h_bottom])
    bbox = [p0[:, np.newaxis], p1[:, np.newaxis], p2[:, np.newaxis], h]
    # import pdb; pdb.set_trace()
    if len(x_initial) == 0:
        x_initial = np.array([center[0], center[1], angle])[:, np.newaxis]

    if use_map_lane:
        lane_dir_vector, confidence = get_lane_direction_api(
            np.array([center[0], center[1]]), R_world, t_world, city_name
        )

        if (
            confidence > 0.85
        ):  # and bbox_min.area < 8 ( (l01_smallest/l02_smallest) > 0.5 and  (l01_smallest/l02_smallest < 2) and bbox_min.area > 5):
            x_initial[2] = np.arctan2(lane_dir_vector[1], lane_dir_vector[0])

            print("Use lane direction!!!!")

    else:

        x_initial[2] = angle

    # import pdb; pdb.set_trace()
    # Use tightest bbox position anyway? that's the way we label gt....
    x_initial[0:2, 0] = center[0:2]

    bbox = transform_bbox(bbox, x_initial)

    return bbox, x_initial


def get_lane_direction_api(pc, R_world, t_world, city_name):

    # city_to_egovehicle_se3 = SE3(rotation=R_world, translation=t_world[:,0])
    # pts = copy.deepcopy(pc)
    # import pdb; pdb.set_trace()
    # pts = city_to_egovehicle_se3.transform_point_cloud(pts)

    lane_dir_vector, confidence = avm.get_lane_direction(pc, city_name, visualize=False)

    # print(pc, lane_dir_vector, confidence)
    # import pdb; pdb.set_trace()
    return lane_dir_vector, confidence


def filter_pc(pc, th=4):

    pc_segs, pc = pc_segmentation_dbscan(
        pc,
        0,
        30,
        0,
        5,
        eps=2.0,
        no_filter=True,
        leaf_size=30,
        remove_ground=False,
        ground_removal_th=0.25,
        remove_plane=False,
    )

    if len(pc_segs) == 0:
        return []

    pc_segs = np.array(pc_segs)
    #
    # pc_segs = pc_segs[0]

    idx_max = 0
    for i in range(len(pc_segs)):

        if len(pc_segs[idx_max]) < len(pc_segs[i]):
            idx_max = i

    pc_max = pc_segs[idx_max]
    if len(pc_max) == 0:
        return []

    center_max = pc_max.sum(axis=0) / len(pc_max)
    for i in range(len(pc_segs)):
        if i == idx_max:
            continue
        c = pc_segs[i].sum(axis=0) / len(pc_segs[i])
        if np.linalg.norm(c - center_max) < th:
            pc_max = np.concatenate((pc_max, pc_segs[i]))

    # import pdb; pdb.set_trace()
    return pc_max[0]


def get_pc_bbox(pc):

    return [
        (pc[:, 0].min(), pc[:, 1].min()),
        (pc[:, 0].max(), pc[:, 1].min()),
        (pc[:, 0].max(), pc[:, 1].max()),
        (pc[:, 0].min(), pc[:, 1].max()),
    ]


def check_pc_overlap(pc1, pc2, min_point_num):
    try:
        b1 = get_pc_bbox(pc1)
        b2 = get_pc_bbox(pc2)

        b1_c = Polygon(b1)
        b2_c = Polygon(b2)
        inter_area = b1_c.intersection(b2_c).area
        union_area = b1_c.area + b2_c.area - inter_area

        if b1_c.area > 11 and b2_c.area > 11:
            overlap = (inter_area / union_area) > 0.5

        elif inter_area > 0:
            overlap = True  # (inter_area/union_area) > 0.5
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

            # not reasonable bbox
            if (
                (area < 2 or area > 12)
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

                # import pdb; pdb.set_trace()
                pc_merged = np.concatenate((pc_merged, pc1[idx_overlap == 0]), axis=0)

    except:
        import pdb

        pdb.set_trace()
    # return overlap, pc_merged
    try:
        if not is_car(pc_merged, min_point_num):
            overlap = False
    except:
        import pdb

        pdb.set_trace()
    return overlap, pc_merged


from operator import itemgetter, attrgetter


def sort_pc_segments_by_area(pc_segments):

    areas = []

    for i in range(len(pc_segments)):

        b1 = get_pc_bbox(pc_segments[i])
        b1_c = Polygon(b1)

        areas.append(b1_c.area)

    return [p for _, p in sorted(zip(areas, pc_segments), key=itemgetter(0))]


def check_pc_bbox_overlap(pc1, bbox2):

    bbox = recover_bounding_box_3d(bbox2)
    b2_c = Polygon(bbox[idx_bbox, :])
    # b2_c = Polygon(np.concatenate((bbox[idx_bbox,:],bbox[idx_bbox[0],:][:,np.newaxis].transpose()),axis=0))
    # import pdb; pdb.set_trace()
    try:
        b1 = get_pc_bbox(pc1)
        # b2 = get_pc_bbox(pc2)
        b1_c = Polygon(b1)
        # b2_c = Polygon(b2)
        inter_area = b1_c.intersection(b2_c).area

        overlap = inter_area > 0

    except:
        import pdb

        pdb.set_trace()
    # print(b1,bbox[idx_bbox,:] , overlap)
    return overlap


def merge_pc(pc1, pc2):
    idx_overlap = np.zeros((len(pc1)))
    for i in range(len(pc1)):
        diff = pc2 - pc1[i]
        diff = np.sum(diff ** 2, axis=1)
        if 0 in diff:
            idx_overlap[i] = 1

    pc_merged = np.concatenate((pc2, pc1[idx_overlap == 0]), axis=0)
    # import pdb; pdb.set_trace()
    return pc_merged


def merge_pcs(pcs, inds):

    pcs = np.array(pcs)
    pcs_out = []
    for i in range(inds.max() + 1):
        inds_selected = inds == i  # overlapping pcs are labeld with same ind

        pcs_selected = pcs[inds_selected]

        pc = pcs_selected[0]
        # import pdb; pdb.set_trace()
        for ii in range(1, len(pcs_selected)):
            # print(ii,len(pc))
            pc = merge_pc(pcs_selected[ii], pc)

        pcs_out.append(pc)

    return pcs_out


import json


def load_lidar_parameters(json_path):
    with open(json_path) as f:
        data = json.load(f)
    # import pdb; pdb.set_trace()

    w_up = np.array(data["vehicle_SE3_up_lidar_"]["rotation"]["coefficients"])
    t_up = np.array(data["vehicle_SE3_up_lidar_"]["translation"])

    w_down = np.array(data["vehicle_SE3_down_lidar_"]["rotation"]["coefficients"])
    t_down = np.array(data["vehicle_SE3_down_lidar_"]["translation"])

    R_avg = Quaternion(w_up).rotation_matrix  # use R_up
    t_avg = (t_up + t_down) / 2

    return -np.matmul(R_avg.transpose(), t_avg[:, np.newaxis]), R_avg.transpose()


import matplotlib.pyplot as plt


def plot_lidar_proj(pc):

    value = np.linalg.norm(pc[:, 0:2], axis=1)
    theta = np.arctan2(pc[:, 1], pc[:, 0]) * 180 / np.pi
    height = np.divide(pc[:, 2], value)
    # colors = np.concatenate((value, value, value), axis=1)
    import pdb

    pdb.set_trace()
    plt.scatter(theta, np.divide(pc[:, 2], value), c=value, s=0.5)
    plt.show()


def pc_project(pc, t, R):
    return (np.matmul(R, pc.transpose()) + t).transpose()


def normalize_theta(theta):  # normalize to -180~180
    theta[theta > 180] -= 360
    theta[theta < -180] += 360
    if (theta < -130).sum() > 0 and (theta > 130).sum() > 0:
        theta += 180
    theta[theta > 180] -= 360
    theta[theta < -180] += 360

    return theta


def get_bbox_pc_project_cylindar(pc, theta_origin, z_max=0.5, z_min=-0.5, scale=200):

    theta = np.arctan2(pc[:, 1], pc[:, 0]) * 180 / np.pi

    theta = normalize_theta(theta)
    theta -= theta_origin
    theta = theta.astype("int")

    if len(theta) > 8:
        theta = normalize_theta(theta)

    theta += 180

    valid_idx = np.logical_and(theta >= 0, theta < 360)

    # if len(theta) > 8:
    #    theta = normalize_theta(theta)

    value = np.linalg.norm(pc[:, 0:2], axis=1)
    height = np.divide(pc[:, 2], value)

    valid_idx = np.logical_and(
        np.logical_and(height > z_min, height < z_max), valid_idx
    )

    # valid_idx = np.logical_and(height>z_min, height<z_max)
    pc = pc[valid_idx]
    value = value[valid_idx]
    height = height[valid_idx]
    theta = theta[valid_idx]
    height = (height - z_min) * scale

    # if valid_idx.sum() != len(theta) and len(theta)==8:
    #    import pdb; pdb.set_trace()
    #    return [],[],[],[]

    if valid_idx.sum() == 0:
        return [], [], [], []

    bbox = []
    #
    # if (theta<50).sum() >0 and (theta>310).sum() > 0 : #boundary case!
    #    idx1 = np.logical_and(theta<180, theta>0)
    #
    #
    #    idx2 = np.logical_and(theta>180, theta<360)
    #
    #    bbox.append(np.array([theta[idx1].min(), theta[idx1].max(), height[idx1].min(), height[idx1].max()]))
    #    bbox.append(np.array([theta[idx2].min(), theta[idx2].max(), height[idx2].min(), height[idx2].max()]))
    #    idx = np.logical_or(idx1, idx2)
    #
    # else:
    idx = np.logical_and(theta >= 0, theta < 360)
    # if idx.sum()==0:
    #    import pdb; pdb.set_trace()
    bbox.append(
        np.array(
            [theta[idx].min(), theta[idx].max(), height[idx].min(), height[idx].max()]
        )
    )

    # if idx.sum() == len(theta):
    #    height = height[idx]
    #    value = value[idx]
    # import pdb; pdb.set_trace()
    return bbox, theta, height, value
    # else:
    #    return [],[],[],[]


def plot_points(xyz):

    pcd = PointCloud()
    pcd.points = Vector3dVector(xyz)

    return pcd


def plot_bbox(points):
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = LineSet()
    line_set.points = Vector3dVector(points)
    line_set.lines = Vector2iVector(lines)
    line_set.colors = Vector3dVector(colors)

    return line_set


def plot_geometry(geo):
    draw_geometries([geo])
