# -*- coding: utf-8 -*-
import numpy as np
import math
from open3d import *

# from mayavi import mlab
import sys

sys.path.append("../../completion/code")
sys.path.append("../../completion/code/utils")
sys.path.append("../../completion/code/utils/bboxmaster")
import utils
from utils import tools
from utils.tools import *

# sys.path.append("../utils/bboxmaster")

from utils.bboxmaster.bbox.bbox3d import BBox3D

import os
import cv2


def get_camera_matrix(p_c, x_c, y_c, z_c, to_print=False):

    ps = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]).transpose(1, 0)

    # import pdb; pdb.set_trace()
    center = np.zeros((1, 3))
    for i in range(4):
        center += ps[:, i]

    center = center / 4  # len(ps)
    ps_c = np.concatenate([p_c + x_c, p_c + y_c, p_c + z_c, p_c], axis=1)

    center_c = np.zeros((1, 3))

    for i in range(4):
        center_c += ps_c[:, i]

    center_c = center_c / 4  # len(ps_c)

    # import pdb; pdb.set_trace()
    H = np.zeros((3, 3))
    for i in range(len(ps[0])):
        H += np.matmul((ps[:, i] - center).transpose(1, 0), (ps_c[:, i] - center_c))

    u, s, vh = np.linalg.svd(H, full_matrices=True)

    if to_print:
        print("center = ", center)
        print("center_c", center_c)
        print(H)
        print(u)
        print(s)
        print(vh)

    V = vh.transpose(1, 0)
    R = np.matmul(V, u.transpose(1, 0))

    if np.linalg.det(R) < 0:  # reflection case
        V[:, 2] *= -1
        R = np.matmul(V, u.transpose(1, 0))

    t = -np.matmul(R, center.transpose(1, 0)) + center_c.transpose(1, 0)

    t = -np.matmul(R.transpose(1, 0), t)
    R = R.transpose(1, 0)

    # if  R[2,2] < 0.34 and R[2,2] > 0.32:
    #    import pdb; pdb.set_trace()

    return R, t


def compute_visible(pc0, R, t, res, step):
    pc = np.matmul(R, pc0.transpose(1, 0)) + t
    pc = pc.transpose(1, 0)
    offset = 5
    ratio = 1

    map_w = math.ceil(
        pc[:, 0].max() - pc[:, 0].min() + offset
    )  # math.ceil(math.sqrt(3) * res)
    map_w = map_w + map_w % 2
    map_h = math.ceil(pc[:, 1].max() - pc[:, 1].min() + offset)
    map_h = map_h + map_h % 2
    map_h = int(map_h * ratio)
    map_w = int(map_w * ratio)

    map_depth = np.zeros((map_h, map_w))
    map_label = -np.ones((map_h, map_w))
    pc_visible = np.zeros((len(pc), 1))

    pc[:, 0] -= pc[:, 0].min()
    pc[:, 1] -= pc[:, 1].min()

    pc0 = pc0[pc[:, 2].argsort(), :]
    pc = pc[pc[:, 2].argsort(), :]

    for i in range(len(pc)):
        u = int(round(pc[i, 0]) * ratio)
        v = int(round(pc[i, 1]) * ratio)

        list_neighbors = []
        for ii in range(-2, 3):
            for jj in range(-2, 3):
                if map_label[v + ii, u + jj] != -1:
                    list_neighbors.append(map_depth[v + ii, u + jj])

        # if len(list_neighbors) > 2:
        # import pdb; pdb.set_trace()

        if map_depth[v, u] == 0:
            map_depth[v, u] = pc[i, 2]
            pc_visible[i] = 1
            map_label[v, u] = i

        elif len(list_neighbors) > 0:
            if min(list_neighbors) >= pc[i, 2]:
                # import pdb; pdb.set_trace()f
                if min(list_neighbors) > pc[i, 2]:
                    pc_visible[int(map_label[v, u])] = 0

                map_label[v, u] = i
                pc_visible[i] = 1

    for i in range(len(pc)):
        u = int(round(pc[i, 0]) * ratio)
        v = int(round(pc[i, 1]) * ratio)

        list_neighbors = []
        for ii in range(-2, 3):
            for jj in range(-2, 3):
                if map_label[v + ii, u + jj] != -1:
                    list_neighbors.append(map_depth[v + ii, u + jj])

        if len(list_neighbors) > 0:
            if min(list_neighbors) < pc[i, 2] and (
                abs(min(list_neighbors) - pc[i, 2]) > 3
            ):
                pc_visible[i] = 0
                # map_label[v,u] = i
            # if min(list_neighbors) >= pc[i,2]:
            #     #import pdb; pdb.set_trace()f
            #     if  min(list_neighbors) > pc[i,2]:
            #         pc_visible[int(map_label[v,u])]=0

            #     map_label[v,u] = i
            #     pc_visible[i]=1

    # compute unoccupied mask

    # import pdb; pdb.set_trace()
    # for i in range(len(pc_all)):
    #     u = int(round(pc_all[i,0])*ratio)
    #     v = int(round(pc_all[i,1])*ratio)

    #     if map_depth[v,u] == 0:
    #         mask[i] = 0

    #     elif map_depth[v,u] < pc_all[i,2]:
    #         mask[i] = 0

    return pc0[np.nonzero(pc_visible == 1)[0], :]


def save_numpy_as_img(m, file_name):
    from PIL import Image

    m = (m - m.min()) / (m.max() - m.min())
    m = m * 255

    im = Image.fromarray(m)
    im = im.convert("RGB")

    im.save(file_name)


def get_pc_normalize_keep_z(
    pc, to_print=False
):  # only need to determine theta and translation(x,y)

    pc_avg = pc.sum(axis=0) / len(pc)
    pc = pc - pc_avg

    pc_2D = np.concatenate((pc[:, 0:2], np.zeros((len(pc), 1))), axis=1)

    cov = np.cov(pc_2D.transpose(1, 0))
    evals, evecs = np.linalg.eig(cov)

    sort_indices = np.argsort(evals)[::-1]
    x_n = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    y_n = evecs[:, sort_indices[1]]
    z_n = evecs[:, sort_indices[2]]

    if x_n[0] < 0:
        x_n = -x_n

    z_test = np.cross(x_n, y_n)
    if z_test[2] < 0:
        y_n = -y_n

    if to_print:
        print(pc_avg)
        print(evals, evecs)
        print(x_n, y_n, z_n)
    # import pdb; pdb.set_trace()

    return get_camera_matrix(
        pc_avg[:, np.newaxis],
        x_n[:, np.newaxis],
        y_n[:, np.newaxis],
        z_n[:, np.newaxis],
        to_print,
    )


def get_z_angle_from_R(R):

    x = np.array([1, 0, 0])[:, np.newaxis]
    x_rotated = np.matmul(R, x)
    theta = np.arctan2(x_rotated, x)

    return theta


def get_pc_normalize(pc):

    pc_avg = pc.sum(axis=0) / len(pc)
    pc = pc - pc_avg

    cov = np.cov(pc.transpose(1, 0))
    evals, evecs = np.linalg.eig(cov)

    sort_indices = np.argsort(evals)[::-1]
    x_n = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    y_n = evecs[:, sort_indices[1]]
    z_n = evecs[:, sort_indices[2]]

    return get_camera_matrix(
        pc_avg[:, np.newaxis],
        x_n[:, np.newaxis],
        y_n[:, np.newaxis],
        z_n[:, np.newaxis],
    )


def get_pc_normalize_small(pc):

    pc_avg = pc.sum(axis=0) / len(pc)
    pc = pc - pc_avg

    cov = np.cov(pc.transpose(1, 0))
    evals, evecs = np.linalg.eig(cov)

    sort_indices = np.argsort(evals)[::0]
    y_n = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_n = evecs[:, sort_indices[1]]
    z_n = evecs[:, sort_indices[2]]

    return get_camera_matrix(
        pc_avg[:, np.newaxis],
        x_n[:, np.newaxis],
        y_n[:, np.newaxis],
        z_n[:, np.newaxis],
    )


def get_bounding_box_3d(
    pc
):  # gives 10 parameters: p0, p1, p2,  p3 = p0 + (p1-p0)x(p2-p0)

    mins = pc.min(axis=0)
    maxs = pc.max(axis=0)

    p0 = mins[:, np.newaxis]
    p1 = np.array([maxs[0], mins[1], mins[2]])[:, np.newaxis]
    p2 = np.array([mins[0], maxs[1], mins[2]])[:, np.newaxis]
    H = maxs[2] - mins[2]

    return [p0, p1, p2, H]


def recover_bounding_box_3d(bbox):
    p0 = bbox[0]
    p1 = bbox[1]
    p2 = bbox[2]
    H = bbox[3]

    delta_p1 = p1 - p0
    delta_p2 = p2 - p0

    p3 = (
        np.cross(
            delta_p1[:, 0] / np.linalg.norm(delta_p1),
            delta_p2[:, 0] / np.linalg.norm(delta_p2),
        )[:, np.newaxis]
        * H
        + p0
    )

    p4 = p0 + delta_p1 + delta_p2
    p5 = p3 + delta_p1
    p6 = p3 + delta_p2
    p7 = p3 + delta_p1 + delta_p2

    return np.concatenate((p0, p1, p2, p3, p4, p5, p6, p7), axis=1).transpose(1, 0)


def transform_bounding_box_3d(bbox, R, t):

    p0 = np.matmul(R, bbox[0]) + t
    p1 = np.matmul(R, bbox[1]) + t
    p2 = np.matmul(R, bbox[2]) + t

    return [p0, p1, p2, bbox[3]]


def save_data_Rt(
    path_folder,
    flag,
    pc,
    pc_norm,
    pc_partial,
    pc_partial_norm,
    bbox,
    bbox_norm,
    R_norm,
    t_norm,
    R_camera,
    t_camera,
    pc_avg,
    mask,
    mask_norm,
):
    print("Write files to ", path_folder)

    path_file = path_folder + "/" + flag + "_pc.ply"
    save_ply(path_file, pc)
    path_file = path_folder + "/" + flag + "_pc_norm.ply"
    save_ply(path_file, pc_norm)

    path_file = path_folder + "/" + flag + "_pc_partial.ply"
    save_ply(path_file, pc_partial)

    path_file = path_folder + "/" + flag + "_pc_partial_norm.ply"
    save_ply(path_file, pc_partial_norm)

    path_file = path_folder + "/" + flag + "_bbox"
    np.save(path_file, bbox)

    path_file = path_folder + "/" + flag + "_bbox_norm"
    np.save(path_file, bbox_norm)

    path_file = path_folder + "/" + flag + "_center"

    center = compute_bbox_center(bbox)

    np.save(path_file, center)

    np.save(path_folder + "/" + flag + "_pc_avg", pc_avg)

    path_file = path_folder + "/" + flag + "_R_camera"
    np.save(path_file, R_camera)
    path_file = path_folder + "/" + flag + "_t_camera"
    np.save(path_file, t_camera)

    path_file = path_folder + "/" + flag + "_R_norm"
    np.save(path_file, R_norm)
    path_file = path_folder + "/" + flag + "_t_norm"
    np.save(path_file, t_norm)

    path_file = path_folder + "/" + flag + "_mask.ply"
    save_ply(path_file, mask)
    path_file = path_folder + "/" + flag + "_mask_norm.ply"
    save_ply(path_file, mask_norm)


def save_data(path_folder, flag, pc, pc_partial, bbox):
    print("Write files to ", path_folder)

    path_file = path_folder + "/" + flag + "_pc.ply"
    save_ply(path_file, pc)
    path_file = path_folder + "/" + flag + "_pc_partial.ply"
    save_ply(path_file, pc_partial)
    ##t
    path_file = path_folder + "/" + flag + "_bbox"
    np.save(path_file, bbox)

    path_file = path_folder + "/" + flag + "_center"

    center = compute_bbox_center(bbox)

    np.save(path_file, center)


def save_ply(path_file, pc):
    # import pdb; pdb.set_trace()
    if isinstance(pc, int):
        print("pc has no value")
        return

    if len(pc) < 10:
        print("Too few points for saving ply!!!!")
        return

    pcd = PointCloud()
    pcd.points = Vector3dVector(pc)
    write_point_cloud(path_file, pcd)


def load_ply(path_file):
    pcd_load = read_point_cloud(path_file)
    xyz_load = np.asarray(pcd_load.points)
    return xyz_load


def compute_bbox_center(bbox):
    p0 = bbox[0]
    p1 = bbox[1]
    p2 = bbox[2]
    H = bbox[3]

    delta_p1 = p1 - p0
    delta_p2 = p2 - p0

    delta_p3 = (
        np.cross(
            delta_p1[:, 0] / np.linalg.norm(delta_p1),
            delta_p2[:, 0] / np.linalg.norm(delta_p2),
        )[:, np.newaxis]
        * H
    )

    return p0 + delta_p1 / 2 + delta_p2 / 2 + delta_p3 / 2


def pc_to_voxel(pc, step, res):  # need to tune step for real data
    # import pdb; pdb.set_trace()
    voxel = np.zeros((res, res, res))

    ori = np.ones((1, 3)) * (-step * res / 2)
    pc = pc - ori
    pc = pc / step
    pc = np.round(pc).astype(int)

    #    pc = keep_bd(pc,res,0)
    pc[pc < 0] = 0
    pc[pc > res - 1] = res - 2

    voxel[pc[:, 0], pc[:, 1], pc[:, 2]] = 1
    #    for i in range(len(pc)):
    #        voxel[keep_bd(pc[i,0],res,0),keep_bd(pc[i,1],res ,0),keep_bd(pc[i,2],res, 0)] = 1

    return voxel


def pc_to_voxel_mask(
    pc, step, res, R_camera, t_camera, R_norm, t_norm, pc_avg, bbox=0, ratio=50
):  # need to tune step for real data
    # compute projection matrix M

    # import pdb; pdb.set_trace()
    voxel = np.zeros((res, res, res))
    voxel_mask = np.zeros((res, res, res))
    pc_avg = pc_avg[:, np.newaxis]
    ori = np.ones((1, 3)) * (-step * res / 2)

    Rm = np.matmul(R_camera, R_norm.transpose(1, 0))
    tm = (
        np.matmul(R_camera, pc_avg - np.matmul(R_norm.transpose(1, 0), t_norm))
        + t_camera
    )

    x_ = np.arange(res)
    y_ = np.arange(res)
    z_ = np.arange(res)

    pv = np.meshgrid(x_, y_, z_)
    pv = np.array(pv).reshape(3, res * res * res)

    pv_proj = np.matmul(Rm, pv * step + ori.transpose(1, 0)) + tm
    pv_proj = pv_proj.transpose(1, 0)
    pv_mask = np.zeros((len(pv_proj), 1))

    # ratio = 50
    offset = 2
    margin = 0.5
    depth_min = 1

    # map_label = -np.ones((map_h,map_w))
    # pc_visible = np.zeros((len(pc),1))

    # pc0 = pc0[pc[:,2].argsort(),:]
    # pc = pc[pc[:,2].argsort(),:]
    # pv[(pv_mask==1)[:,0],:]
    # import pdb; pdb.set_trace()

    eps = 0.0000001

    pc = pc - ori
    pc = pc / step
    pc = np.round(pc).astype(int)
    pc_proj = np.matmul(Rm, pc.transpose(1, 0) * step + ori.transpose(1, 0)) + tm

    pc_proj = pc_proj.transpose(1, 0)

    pv_proj[:, 2] += eps
    pv_proj[:, 0] = pv_proj[:, 0] / pv_proj[:, 2]
    pv_proj[:, 1] = pv_proj[:, 1] / pv_proj[:, 2]
    pv = pv.transpose(1, 0)
    pv = pv[pv_proj[:, 2] > depth_min, :]

    pv_proj = pv_proj[pv_proj[:, 2] > depth_min, :]

    pc_proj[:, 0] = pc_proj[:, 0] / pc_proj[:, 2]
    pc_proj[:, 1] = pc_proj[:, 1] / pc_proj[:, 2]

    pc_proj[:, 0] -= pv_proj[:, 0].min()
    pc_proj[:, 1] -= pv_proj[:, 1].min()

    pv_proj[:, 0] -= pv_proj[:, 0].min()
    pv_proj[:, 1] -= pv_proj[:, 1].min()

    map_w = math.ceil(
        pv_proj[:, 0].max() * ratio - pv_proj[:, 0].min() * ratio + offset
    )  # math.ceil(math.sqrt(3) * res)
    map_w = map_w + map_w % 2
    map_h = math.ceil(
        pv_proj[:, 1].max() * ratio - pv_proj[:, 1].min() * ratio + offset
    )
    map_h = map_h + map_h % 2
    map_h = int(map_h)
    map_w = int(map_w)

    map_depth = np.zeros((map_h, map_w))

    u = np.round(pc_proj[:, 0] * ratio)
    v = np.round(pc_proj[:, 1] * ratio)
    u[u > map_h - 1] = map_h - 1
    v[v > map_w - 1] = map_w - 1
    u[u < 0] = 0
    v[v < 0] = 0
    # import pdb; pdb.set_trace()
    map_depth[v.astype("int"), u.astype("int")] = pc_proj[:, 2]

    #    for i in range(len(pc_proj)):
    #        u = int(round(pc_proj[i,0]*ratio))
    #        v = int(round(pc_proj[i,1]*ratio))
    #        map_depth[keep_bd(v,map_h,0), keep_bd(u, map_w,0)] = pc_proj[i,2]

    # import scipy.misc
    # scipy.misc.toimage(map_depth, cmin=map_depth.min(), cmax=map_depth.max()).save('test.png')
    # import pdb; pdb.set_trace()

    pv = pv[pv_proj[:, 2].argsort(), :]
    pv_proj = pv_proj[pv_proj[:, 2].argsort(), :]

    for i in range(len(pv_proj)):
        u = int(round(pv_proj[i, 0] * ratio))
        v = int(round(pv_proj[i, 1] * ratio))

        list_neighbors = []
        for ii in range(-2, 3):
            for jj in range(-2, 3):
                if map_depth[keep_bd(v + ii, map_h, 0), keep_bd(u + jj, map_w, 0)] > 0:
                    list_neighbors.append(
                        map_depth[keep_bd(v + ii, map_h, 0), keep_bd(u + jj, map_w, 0)]
                    )

        # if len(list_neighbors) > 0:
        #     if min(list_neighbors) >= pc[i,2]:
        #         #import pdb; pdb.set_trace()f
        #         if  min(list_neighbors) > pc[i,2]:
        #             pc_visible[int(map_label[v,u])]=0

        if map_depth[v, u] == 0:
            pv_mask[i] = 0
        elif len(list_neighbors) > 0:
            if min(list_neighbors) - margin > pv_proj[i, 2]:
                # import pdb; pdb.set_trace()f
                pv_mask[i] = 1

        # if map_depth[v,u] == 0:
        #     pv_mask[i] = 0
        # elif map_depth[v,u] - margin > pv_proj[i,2]:
        #    load_mask[i] = 1

    pc[pc > res - 1] = res
    pc[pc < 0] = 0
    voxel[pc[:, 0], pc[:, 1], pc[:, 2]] = 1
    #    for i in range(len(pc)):
    #        voxel[keep_bd(pc[i,0],res,0),keep_bd(pc[i,1],res ,0),keep_bd(pc[i,2],res, 0)] = 1

    # pv = pv[(pv_mask==1)[:,0],:]
    # import pdb; pdb.set_trace()
    # bbox[0] = (bbox[0] - ori.transpose(1,0))/step
    # bbox[1] = (bbox[1] - ori.transpose(1,0))/step
    # bbox[2] = (bbox[2] - ori.transpose(1,0))/step
    # bbox[3] = bbox[3]/step

    # point_bbox = recover_bounding_box_3d(bbox)
    # u = point_bbox[1,:] - point_bbox[0,:]
    # v = point_bbox[2,:] - point_bbox[0,:]
    # w = point_bbox[3,:] - point_bbox[0,:]

    # u_p0 = np.dot(u, point_bbox[0,:])
    # v_p0 = np.dot(v, point_bbox[0,:])
    # w_p0 = np.dot(w, point_bbox[0,:])
    # u_p1 = np.dot(u, point_bbox[1,:])
    # v_p2 = np.dot(v, point_bbox[2,:])
    # w_p3 = np.dot(w, point_bbox[3,:])

    for i in range(len(pv)):
        # u_p = np.dot(pv[i,:], u).sum()
        # v_p = np.dot(pv[i,:], v).sum()
        # w_p = np.dot(pv[i,:], w).sum()

        # if not ( in_between(u_p, u_p0, u_p1, margin) and in_between(v_p, v_p0, v_p2, margin) and in_between(w_p, w_p0, w_p3, margin)):
        #    voxel_mask[keep_bd(pv[i,0],res,0),keep_bd(pv[i,1],res ,0),keep_bd(pv[i,2],res, 0)] = 1

        if pv_mask[i, 0] == 1:
            voxel_mask[
                keep_bd(pv[i, 0], res, 0),
                keep_bd(pv[i, 1], res, 0),
                keep_bd(pv[i, 2], res, 0),
            ] = 1

    # for i in range()

    #     list_neighbors = []
    #     for ii in range(-2,3):
    #         for jj in range(-2,3):
    #             if map_label[v+ii,u+jj] != -1:
    #                 list_neighbors.append(map_depth[v+ii,u+jj])

    #     # if len(list_neighbors) > 2:
    #     #import pdb; pdb.set_trace()

    #     if map_depth[v,u] == 0:
    #         map_depth[v,u] = pc[i,2]
    #         pc_visible[i]=1
    #         map_label[v,u] = i

    #     elif len(list_neighbors) > 0:
    #         if min(list_neighbors) >= pc[i,2]:
    #             #import pdb; pdb.set_trace()f
    #             if  min(list_neighbors) > pc[i,2]:
    #                 pc_visible[int(map_label[v,u])]=0

    #             map_label[v,u] = i
    #             pc_visible[i]=1

    # for i in range(len(pc)):
    #     u = int(round(pc[i,0])*ratio)
    #     v = int(round(pc[i,1])*ratio)

    #     list_neighbors = []
    #     for ii in range(-2,3):
    #         for jj in range(-2,3):
    #             if map_label[v+ii,u+jj] != -1:
    #                 list_neighbors.append(map_depth[v+ii,u+jj])

    #     if len(list_neighbors) > 0:
    #         if  min(list_neighbors) < pc[i,2] and (abs(min(list_neighbors) - pc[i,2]) > 3):
    #             pc_visible[i]=0

    # voxel = np.zeros((res,res,res))

    # ori = np.ones((1,3)) * (-step*res/2)
    # pc = pc - ori
    # pc = pc/step
    # pc = np.round(pc).astype(int)
    # for i in range(len(pc)):
    #     voxel[keep_bd(pc[i,0],res,0),keep_bd(pc[i,1],res ,0),keep_bd(pc[i,2],res, 0)] = 1

    return voxel, voxel_mask


def voxel_to_pc(v, step, res, th=0.5):
    # import pdb; pdb.set_trace()
    ori = np.ones((1, 3)) * (-step * res / 2)
    pc = []
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            for k in range(v.shape[2]):
                if v[i, j, k] > th:
                    pc.append(np.array([i, j, k]).astype("float") * step + ori)

    pc = np.array(pc).reshape(len(pc), 3)

    return pc


def bbox_center_to_voxel(bbox, step, res):
    ori = np.ones((3, 1)) * (-step * res / 2)
    for i in range(3):
        bbox[i] = (bbox[i] - ori) / step

        bbox[i][0] = keep_bd(bbox[i][0], res, 0)
        bbox[i][1] = keep_bd(bbox[i][1], res, 0)
        bbox[i][2] = keep_bd(bbox[i][2], res, 0)

    bbox[-1] /= step

    center = compute_bbox_center(bbox)
    return bbox, center


def bbox_center_to_pc(bbox, step, res, scale):
    ori = np.ones((3, 1)) * (-step * res / 2)
    for i in range(3):
        bbox[i] = bbox[i] * step * scale + ori

    bbox[-1] = bbox[-1] * step * scale
    center = compute_bbox_center(bbox)
    return bbox, center


def load_npz(path_npz):
    return np.load(path_npz)


def check_bd(v, upper, lower):
    if v < upper and v >= lower:
        return True

    return False


def keep_bd(v, upper, lower):
    if v < upper and v >= lower:
        return v

    elif v >= upper - 1:
        return upper - 1

    return lower


def keep_bd_include(v, upper, lower):
    if v < upper and v >= lower:
        return v

    elif v >= upper:
        return upper

    return lower


def array_normalize(m):
    if m.size == 0:
        return m

    m = m - m.min()
    m = m / m.max() * 1

    return m


def plot_3d_sdf(sdf, plot_pos=True):
    # mlab.figure(bgcolor=(1.0,1.0,1.0))
    # x,y,z=np.meshgrid(np.arange(0,sdf.shape[0]),np.arange(0,sdf.shape[1]),np.ones((1,sdf.shape[2]))*12)#np.arange(0,sdf.shape[2]))
    x, y, z = np.meshgrid(
        np.arange(0, sdf.shape[0]),
        np.arange(0, sdf.shape[1]),
        np.arange(0, sdf.shape[2]),
    )  # np.arange(0,sdf.shape[2]))

    x = x.transpose((1, 0, 2))
    y = y.transpose((1, 0, 2))
    z = z.transpose((1, 0, 2))

    # cut=12
    # x = x[cut,:,:]
    # y = y[cut,:,:]
    # z = z[cut,:,:]
    # sdf = sdf[cut,:,:]
    # x =     x[:,:,cut]
    # y =     y[:,:,cut]
    # z =     z[:,:,cut]
    # sdf = sdf[:,:,cut]
    # # x =     x[:,cut,:]
    # # y =     y[:,cut,:]
    # # z =     z[:,cut,:]
    # # sdf = sdf[:,cut,:]

    sdf1 = array_normalize(sdf[sdf > 0])
    sdf2 = array_normalize(sdf[sdf == 0])
    sdf3 = array_normalize(-(sdf[sdf < 0]))

    if sdf1.size > 0 and plot_pos:
        nodes1 = mlab.points3d(
            x[sdf > 0],
            y[sdf > 0],
            z[sdf > 0],
            sdf1,
            scale_factor=1,
            colormap="Blues",
            opacity=0.3,
        )
        nodes1.glyph.scale_mode = "scale_by_vector"

    if sdf2.size > 0:
        nodes2 = mlab.points3d(
            x[sdf == 0],
            y[sdf == 0],
            z[sdf == 0],
            sdf2,
            scale_factor=1,
            color=(0.5, 0.5, 0.5),
            opacity=0.3,
        )
        nodes2.glyph.scale_mode = "scale_by_vector"

    if sdf3.size > 0:
        nodes3 = mlab.points3d(
            x[sdf < 0],
            y[sdf < 0],
            z[sdf < 0],
            sdf3,
            scale_factor=1,
            colormap="Reds",
            opacity=0.3,
        )
        nodes3.glyph.scale_mode = "scale_by_vector"

    # mport pdb;pdb.set_trace()


def plot_3d_bbox(
    pc,
    pc_partial,
    mask,
    bbox,
    center,
    plot_bbox=True,
    plot_pc=True,
    plot_pc_partial=True,
    plot_axis=True,
    plot_mask=True,
):

    # mlab.figure(bgcolor=(1.0,1.0,1.0))
    # mlab.points3d(center[0], center[1], center[2],0.1,scale_factor=2, color=(1, 0.3, 0.3))

    if plot_bbox:
        bbox_linewidth = 0.1  # 0.01
        bbox_color = (1.0, 0.8, 0.6)
        point_bbox = recover_bounding_box_3d(bbox)
        point_bbox = np.array(point_bbox)

        mlab.plot3d(
            point_bbox[0:2, 0],
            point_bbox[0:2, 1],
            point_bbox[0:2, 2],
            tube_radius=bbox_linewidth * 2,
            color=bbox_color,
            opacity=0.3,
        )
        mlab.plot3d(
            point_bbox[[0, 2], 0],
            point_bbox[[0, 2], 1],
            point_bbox[[0, 2], 2],
            tube_radius=bbox_linewidth * 2,
            color=bbox_color,
            opacity=0.3,
        )
        mlab.plot3d(
            point_bbox[[0, 3], 0],
            point_bbox[[0, 3], 1],
            point_bbox[[0, 3], 2],
            tube_radius=bbox_linewidth * 2,
            color=bbox_color,
            opacity=0.3,
        )
        mlab.plot3d(
            point_bbox[[4, 1], 0],
            point_bbox[[4, 1], 1],
            point_bbox[[4, 1], 2],
            tube_radius=bbox_linewidth,
            color=bbox_color,
            opacity=0.3,
        )
        mlab.plot3d(
            point_bbox[[4, 2], 0],
            point_bbox[[4, 2], 1],
            point_bbox[[4, 2], 2],
            tube_radius=bbox_linewidth,
            color=bbox_color,
            opacity=0.3,
        )
        mlab.plot3d(
            point_bbox[[4, 7], 0],
            point_bbox[[4, 7], 1],
            point_bbox[[4, 7], 2],
            tube_radius=bbox_linewidth,
            color=bbox_color,
            opacity=0.3,
        )
        mlab.plot3d(
            point_bbox[[6, 2], 0],
            point_bbox[[6, 2], 1],
            point_bbox[[6, 2], 2],
            tube_radius=bbox_linewidth,
            color=bbox_color,
            opacity=0.3,
        )
        mlab.plot3d(
            point_bbox[[6, 3], 0],
            point_bbox[[6, 3], 1],
            point_bbox[[6, 3], 2],
            tube_radius=bbox_linewidth,
            color=bbox_color,
            opacity=0.3,
        )
        mlab.plot3d(
            point_bbox[[6, 7], 0],
            point_bbox[[6, 7], 1],
            point_bbox[[6, 7], 2],
            tube_radius=bbox_linewidth,
            color=bbox_color,
            opacity=0.3,
        )
        mlab.plot3d(
            point_bbox[[5, 1], 0],
            point_bbox[[5, 1], 1],
            point_bbox[[5, 1], 2],
            tube_radius=bbox_linewidth,
            color=bbox_color,
            opacity=0.3,
        )
        mlab.plot3d(
            point_bbox[[5, 3], 0],
            point_bbox[[5, 3], 1],
            point_bbox[[5, 3], 2],
            tube_radius=bbox_linewidth,
            color=bbox_color,
            opacity=0.3,
        )
        mlab.plot3d(
            point_bbox[[5, 7], 0],
            point_bbox[[5, 7], 1],
            point_bbox[[5, 7], 2],
            tube_radius=bbox_linewidth,
            color=bbox_color,
            opacity=0.3,
        )

    if plot_axis:
        # import pdb;pdb.set_trace()
        length_axis = 25.0
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

    if plot_pc:
        # import pdb; pdb.set_trace()
        nodes = mlab.points3d(
            pc[:, 0],
            pc[:, 1],
            pc[:, 2],
            np.square(pc[:, 0]) + np.square(pc[:, 1]) * 20 + 30,
            scale_factor=0.9,
            colormap="gist_gray",
            opacity=1.0,
        )
        nodes.glyph.scale_mode = "scale_by_vector"
    if plot_pc_partial:
        nodes2 = mlab.points3d(
            pc_partial[:, 0],
            pc_partial[:, 1],
            pc_partial[:, 2],
            scale_factor=1,
            color=(1.0, 0.6, 0.8),
        )
        nodes2.glyph.scale_mode = "scale_by_vector"

    if plot_mask:
        # import pdb; pdb.set_trace()
        nodes = mlab.points3d(
            mask[:, 0],
            mask[:, 1],
            mask[:, 2],
            np.square(mask[:, 0]) + np.square(mask[:, 1]) * 10 + 30,
            scale_factor=0.9,
            color=(0.8, 0.8, 0.9),
            opacity=0.3,
        )
        nodes.glyph.scale_mode = "scale_by_vector"


def plot_3d_bbox_figure1(
    pc,
    pc_partial,
    bbox,
    center,
    plot_bbox=True,
    plot_pc=True,
    plot_pc_partial=True,
    plot_axis=True,
):

    # mlab.figure(bgcolor=(1.0,1.0,1.0))
    # mlab.points3d(center[0], center[1], center[2],0.1,scale_factor=2, color=(1, 0.3, 0.3))
    if plot_axis:
        # import pdb;pdb.set_trace()
        length_axis = 5.0
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

    if plot_pc:
        # import pdb; pdb.set_trace()
        nodes = mlab.points3d(
            pc[:, 0],
            pc[:, 1],
            pc[:, 2],
            np.square(pc[:, 0]) + np.square(pc[:, 1]) * 20 + 30,
            scale_factor=0.1,
            colormap="gist_gray",
            opacity=0.3,
        )
        nodes.glyph.scale_mode = "scale_by_vector"
    if plot_pc_partial:
        nodes2 = mlab.points3d(
            pc_partial[:, 0],
            pc_partial[:, 1],
            pc_partial[:, 2],
            scale_factor=0.2,
            color=(1.0, 0.6, 0.8),
        )
        nodes2.glyph.scale_mode = "scale_by_vector"

    if plot_bbox:
        bbox_linewidth = 0.08  # 0.01
        bbox_color = (1.0, 0.8, 0.6)
        point_bbox = recover_bounding_box_3d(bbox)
        point_bbox = np.array(point_bbox)

        mlab.plot3d(
            point_bbox[0:2, 0],
            point_bbox[0:2, 1],
            point_bbox[0:2, 2],
            tube_radius=bbox_linewidth * 2,
            color=bbox_color,
            opacity=0.5,
        )
        mlab.plot3d(
            point_bbox[[0, 2], 0],
            point_bbox[[0, 2], 1],
            point_bbox[[0, 2], 2],
            tube_radius=bbox_linewidth * 2,
            color=bbox_color,
            opacity=0.5,
        )
        mlab.plot3d(
            point_bbox[[0, 3], 0],
            point_bbox[[0, 3], 1],
            point_bbox[[0, 3], 2],
            tube_radius=bbox_linewidth * 2,
            color=bbox_color,
            opacity=0.5,
        )
        mlab.plot3d(
            point_bbox[[4, 1], 0],
            point_bbox[[4, 1], 1],
            point_bbox[[4, 1], 2],
            tube_radius=bbox_linewidth,
            color=bbox_color,
            opacity=0.5,
        )
        mlab.plot3d(
            point_bbox[[4, 2], 0],
            point_bbox[[4, 2], 1],
            point_bbox[[4, 2], 2],
            tube_radius=bbox_linewidth,
            color=bbox_color,
            opacity=0.5,
        )
        mlab.plot3d(
            point_bbox[[4, 7], 0],
            point_bbox[[4, 7], 1],
            point_bbox[[4, 7], 2],
            tube_radius=bbox_linewidth,
            color=bbox_color,
            opacity=0.5,
        )
        mlab.plot3d(
            point_bbox[[6, 2], 0],
            point_bbox[[6, 2], 1],
            point_bbox[[6, 2], 2],
            tube_radius=bbox_linewidth,
            color=bbox_color,
            opacity=0.5,
        )
        mlab.plot3d(
            point_bbox[[6, 3], 0],
            point_bbox[[6, 3], 1],
            point_bbox[[6, 3], 2],
            tube_radius=bbox_linewidth,
            color=bbox_color,
            opacity=0.5,
        )
        mlab.plot3d(
            point_bbox[[6, 7], 0],
            point_bbox[[6, 7], 1],
            point_bbox[[6, 7], 2],
            tube_radius=bbox_linewidth,
            color=bbox_color,
            opacity=0.5,
        )
        mlab.plot3d(
            point_bbox[[5, 1], 0],
            point_bbox[[5, 1], 1],
            point_bbox[[5, 1], 2],
            tube_radius=bbox_linewidth,
            color=bbox_color,
            opacity=0.5,
        )
        mlab.plot3d(
            point_bbox[[5, 3], 0],
            point_bbox[[5, 3], 1],
            point_bbox[[5, 3], 2],
            tube_radius=bbox_linewidth,
            color=bbox_color,
            opacity=0.5,
        )
        mlab.plot3d(
            point_bbox[[5, 7], 0],
            point_bbox[[5, 7], 1],
            point_bbox[[5, 7], 2],
            tube_radius=bbox_linewidth,
            color=bbox_color,
            opacity=0.5,
        )


def bbox_from_flat(bbox):
    p0 = bbox[0:3][:, np.newaxis]
    p1 = bbox[3:6][:, np.newaxis]
    p2 = bbox[6:9][:, np.newaxis]
    return [p0, p1, p2, bbox[-1]]


def apply_thetaz_t(pc, theta_z, t):

    pc_avg = pc.sum(axis=0) / len(pc)
    pc = pc - pc_avg
    R = np.array(
        [
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1],
        ]
    )

    return (np.matmul(R, pc.transpose(1, 0)) + t).transpose(1, 0)


def compute_theta_z(R):

    tmp = (np.trace(R) - 1) / 2
    theta = np.arccos(keep_bd_include(tmp, 1, -1))

    # x = (m21 - m12)/√((m21 - m12)2+(m02 - m20)2+(m10 - m01)2)
    # y = (m02 - m20)/√((m21 - m12)2+(m02 - m20)2+(m10 - m01)2)
    if (R[1, 0] - R[0, 1]) / (2 * np.sin(theta) + 0.0000001) < 0:
        theta = -theta

    return theta


def get_pc_from_cuboid(
    pc_raw, cuboids, imgs, bboxs_crop, R_world, t_world, crop_image=True
):
    pc_cuboids = []  # [None] * U.shape[0]
    bboxs = []  # [None] * U.shape[0]

    pc_cuboids_norm = []  # [None] * U.shape[0]
    bboxs_norm = []  # [None] * U.shape[0]
    R_norms = []  # [None] * U.shape[0]
    t_norms = []  # [None] * U.shape[0]
    pc_avgs = []  # [None] * U.shape[0]

    pc_cuboids_global = []  # [None] * U.shape[0]
    bboxs_global = []  # [None] * U.shape[0]
    R_norms_global = []  # [None] * U.shape[0]
    t_norms_global = []  # [None] * U.shape[0]
    pc_avgs_global = []  # [None] * U.shape[0]

    img_crops = []
    R_cameras = []
    t_cameras = []
    p_world = []

    if len(pc_raw) < 100 or len(cuboids) == 0:
        return (
            pc_cuboids,
            bboxs,
            pc_cuboids_norm,
            bboxs_norm,
            R_norms,
            t_norms,
            pc_avgs,
            pc_cuboids_global,
            bboxs_global,
            R_norms_global,
            t_norms_global,
            pc_avgs_global,
            img_crops,
            R_cameras,
            t_cameras,
            p_world,
        )
    U = []
    V = []
    W = []
    P1 = []
    P2 = []
    P4 = []
    P5 = []
    index = []
    for c in range(len(cuboids)):
        if not "center" in cuboids[c]:
            continue
        bbox = BBox3D(
            cuboids[c]["center"]["x"],
            cuboids[c]["center"]["y"],
            cuboids[c]["center"]["z"],
            cuboids[c]["dimensions"]["length"],
            cuboids[c]["dimensions"]["width"],
            cuboids[c]["dimensions"]["height"],
            rx=cuboids[c]["rotation"]["x"],
            ry=cuboids[c]["rotation"]["y"],
            rz=cuboids[c]["rotation"]["z"],
            rw=cuboids[c]["rotation"]["w"],
        )

        u = bbox.p2 - bbox.p1
        v = bbox.p4 - bbox.p1
        w = bbox.p5 - bbox.p1

        U.append(u[0:3])
        V.append(v[0:3])
        W.append(w[0:3])
        P1.append(bbox.p1[0:3])
        P2.append(bbox.p2[0:3])
        P4.append(bbox.p4[0:3])
        P5.append(bbox.p5[0:3])
        index.append(c)

    if len(U) == 0:
        return (
            pc_cuboids,
            bboxs,
            pc_cuboids_norm,
            bboxs_norm,
            R_norms,
            t_norms,
            pc_avgs,
            pc_cuboids_global,
            bboxs_global,
            R_norms_global,
            t_norms_global,
            pc_avgs_global,
            img_crops,
            R_cameras,
            t_cameras,
            p_world,
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

    for c in range(U.shape[0]):
        p = pc_raw[flag[c, :]]
        # import pdb; pdb.set_trace()

        if np.isnan(p).any() or np.isinf(p).any() or len(p) < 50:
            # print('NAN or INF detected!! Or point cloud contains too few points')
            # import pdb; pdb.set_trace()
            continue

        # try:

        # except:
        #    import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        p_world.append(np.matmul(R_world, p.transpose(1, 0)).transpose(1, 0) + t_world)

        pc_avg_global = p.sum(axis=0) / len(p)
        p = p - pc_avg_global
        R_norm_global, t_norm_global = get_pc_normalize_keep_z(p)
        p_norm = np.matmul(R_norm_global, p.transpose(1, 0)) + t_norm_global

        p0_norm = (
            np.matmul(R_norm_global, (P1[c, :] - pc_avg_global)[:, np.newaxis])
            + t_norm_global
        )
        p1_norm = (
            np.matmul(R_norm_global, (P2[c, :] - pc_avg_global)[:, np.newaxis])
            + t_norm_global
        )
        p2_norm = (
            np.matmul(R_norm_global, (P4[c, :] - pc_avg_global)[:, np.newaxis])
            + t_norm_global
        )

        if p0_norm[0] > 0:  # align car head to x-axis!!!
            R_pi = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            p_pi = np.matmul(R_pi, p.transpose(1, 0))
            R_norm_global, t_norm_global = get_pc_normalize_keep_z(p_pi.transpose(1, 0))

            R_norm_global = np.matmul(R_norm_global, R_pi)
            p_norm = np.matmul(R_norm_global, p.transpose(1, 0)) + t_norm_global

            p0_norm = (
                np.matmul(R_norm_global, (P1[c, :] - pc_avg_global)[:, np.newaxis])
                + t_norm_global
            )
            p1_norm = (
                np.matmul(R_norm_global, (P2[c, :] - pc_avg_global)[:, np.newaxis])
                + t_norm_global
            )
            p2_norm = (
                np.matmul(R_norm_global, (P4[c, :] - pc_avg_global)[:, np.newaxis])
                + t_norm_global
            )

        h = np.linalg.norm(P5[c, :] - P1[c, :])
        bbox_norm = [p0_norm, p1_norm, p2_norm, h]

        # import pdb; pdb.set_trace()
        x_norm = p1_norm - p0_norm
        x_norm[2] = 0
        x_norm = x_norm / np.linalg.norm(x_norm)
        x_axis = np.array([1, 0, 0])[:, np.newaxis]

        vv = np.cross(x_axis[:, 0], x_norm[:, 0])
        # ss = np.linalg.norm(vv)
        cc = np.dot(x_norm[:, 0], x_axis[:, 0])

        vx = np.array([[0, -vv[2], vv[1]], [vv[2], 0, -vv[0]], [-vv[1], vv[0], 0]])

        if cc == -1:
            R_norm = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        else:
            R_norm = np.eye(3) + vx + np.dot(vx, vx) * 1 / (1 + cc)

        t_norm = compute_bbox_center(bbox_norm)
        # import pdb;pdb.set_trace()

        R_norm_inv = R_norm.transpose(1, 0)
        t_norm_inv = -np.matmul(R_norm.transpose(1, 0), t_norm)
        pc = np.matmul(R_norm_inv, p_norm) + t_norm_inv

        pc_avg = pc.transpose(1, 0).sum(axis=0) / len(pc[0])
        # R_norm, t_norm2 = get_pc_normalize_keep_z(pc.transpose(1,0)-pc_avg)

        R_norm_inv = R_norm.transpose(1, 0)
        t_norm_inv = -np.matmul(R_norm.transpose(1, 0), t_norm)

        p0 = np.matmul(R_norm_inv, p0_norm) + t_norm_inv + pc_avg[:, np.newaxis]
        p1 = np.matmul(R_norm_inv, p1_norm) + t_norm_inv + pc_avg[:, np.newaxis]
        p2 = np.matmul(R_norm_inv, p2_norm) + t_norm_inv + pc_avg[:, np.newaxis]

        # bbox = np.concatenate((p0_norm.transpose(1,0), p1_norm.transpose(1,0), p2_norm.transpose(1,0),np.array(h).reshape(1,1)),axis=1)

        bbox_global = [
            P1[c, :][:, np.newaxis],
            P2[c, :][:, np.newaxis],
            P4[c, :][:, np.newaxis],
            h,
        ]
        bbox = [p0, p1, p2, h]
        # import pdb;pdb.set_trace()

        pc_cuboids.append(pc.transpose(1, 0))
        bboxs.append(bbox)

        pc_cuboids_norm.append(p_norm.transpose(1, 0))
        bboxs_norm.append(bbox_norm)
        R_norms.append(R_norm)
        t_norms.append(t_norm)
        pc_avgs.append(pc_avg)

        pc_cuboids_global.append(pc_raw[flag[c, :]])
        bboxs_global.append(bbox_global)
        R_norms_global.append(R_norm_global)
        t_norms_global.append(t_norm_global)
        pc_avgs_global.append(pc_avg_global)

        # bboxs.append(bbox)
        # R_norms.append(R_norm)
        # t_norms.append(t_norm)
        # pc_avgs.append(pc_avg)
        # pc_cuboids.append(p_norm.transpose(1,0))

        if crop_image:
            bbox_crop = bboxs_crop[index[c]]
            height, width, channels = imgs[index[c]].shape

            # import pdb; pdb.set_trace()
            img_crops.append(
                imgs[index[c]][
                    int(bbox_crop[1] * height) : int(bbox_crop[3] * height),
                    int(bbox_crop[0] * width) : int(bbox_crop[2] * width),
                ]
            )

        # import pdb; pdb.set_trace()
        p_c = -pc_avg_global
        p_c = np.matmul(R_norm_global, p_c[:, np.newaxis]) + t_norm_global

        z_c = -p_c / np.linalg.norm(p_c)
        x_c = np.array([-z_c[1][0], z_c[0][0], 0])
        x_c = x_c / np.linalg.norm(x_c)

        x_c = x_c[:, np.newaxis]

        y_c = np.cross(z_c[:, 0], x_c[:, 0])
        if y_c[2] > 0:
            y_c = -y_c
            x_c = -x_c

        y_c = y_c[:, np.newaxis]

        R_camera, t_camera = get_camera_matrix(p_c, x_c, y_c, z_c)
        R_cameras.append(R_camera)
        t_cameras.append(t_camera)

    #    import pdb; pdb.set_trace()
    return (
        pc_cuboids,
        bboxs,
        pc_cuboids_norm,
        bboxs_norm,
        R_norms,
        t_norms,
        pc_avgs,
        pc_cuboids_global,
        bboxs_global,
        R_norms_global,
        t_norms_global,
        pc_avgs_global,
        img_crops,
        R_cameras,
        t_cameras,
        p_world,
    )


def save_plys(
    pc_cuboids,
    bboxs,
    pc_cuboids_norm,
    bboxs_norm,
    R_norms,
    t_norms,
    pc_avgs,
    pc_cuboids_global,
    bboxs_global,
    R_norms_global,
    t_norms_global,
    pc_avgs_global,
    img_crops,
    R_cameras,
    t_cameras,
    R_world,
    t_world,
    pc_world,
    path_plys,
    path_folder,
    car_labels,
    flag,
    save_image=True,
):
    #%todo: save bbox gts
    #%decide bbox format!!!!

    if not os.path.exists(path_folder):
        os.makedirs(path_folder)
    for i in range(len(pc_cuboids)):

        path_subfolder = os.path.join(path_folder, car_labels[i])
        if not os.path.isdir(path_subfolder):
            os.mkdir(path_subfolder)

        save_ply(
            path_subfolder + "/" + flag + "_" + str(i) + "_pc_world.ply", pc_world[i]
        )
        save_ply(
            path_subfolder + "/" + flag + "_" + str(i) + "_pc_partial_norm.ply",
            pc_cuboids_norm[i],
        )
        save_ply(
            path_subfolder + "/" + flag + "_" + str(i) + "_pc_partial_global.ply",
            pc_cuboids_global[i],
        )
        save_ply(
            path_subfolder + "/" + flag + "_" + str(i) + "_pc_partial.ply",
            pc_cuboids[i],
        )

        np.save(path_subfolder + "/" + flag + "_" + str(i) + "_bbox.npy", bboxs[i])
        np.save(
            path_subfolder + "/" + flag + "_" + str(i) + "_bbox_norm.npy", bboxs_norm[i]
        )
        np.save(path_subfolder + "/" + flag + "_" + str(i) + "_R_norm.npy", R_norms[i])
        np.save(path_subfolder + "/" + flag + "_" + str(i) + "_t_norm.npy", t_norms[i])
        np.save(path_subfolder + "/" + flag + "_" + str(i) + "_pc_avg.npy", pc_avgs[i])

        np.save(
            path_subfolder + "/" + flag + "_" + str(i) + "_bbox_global.npy",
            bboxs_global[i],
        )
        np.save(
            path_subfolder + "/" + flag + "_" + str(i) + "_R_norm_global.npy",
            R_norms_global[i],
        )
        np.save(
            path_subfolder + "/" + flag + "_" + str(i) + "_t_norm_global.npy",
            t_norms_global[i],
        )
        np.save(
            path_subfolder + "/" + flag + "_" + str(i) + "_pc_avg_global.npy",
            pc_avgs_global[i],
        )

        np.save(
            path_subfolder + "/" + flag + "_" + str(i) + "_R_camera.npy", R_cameras[i]
        )
        np.save(
            path_subfolder + "/" + flag + "_" + str(i) + "_t_camera.npy", t_cameras[i]
        )
        np.save(path_subfolder + "/" + flag + "_" + str(i) + "_R_world.npy", R_world)
        np.save(path_subfolder + "/" + flag + "_" + str(i) + "_t_world.npy", t_world)

        if save_image:
            cv2.imwrite(
                path_subfolder + "/" + flag + "_" + str(i) + "_im_crop.png",
                img_crops[i],
            )

        f = open(path_subfolder + "/" + flag + "_" + str(i) + "_global_ply.txt", "w")
        f.write(path_plys[0])
        f.close()

    #     #import pdb; pdb.set_trace()
    #     for c in range(len(cuboids)):
    #         if not 'center' in cuboids[c]:
    #             continue

    #         bbox = BBox3D(cuboids[c]['center']['x'], \
    #             cuboids[c]['center']['y'], \
    #             cuboids[c]['center']['z'], \
    #             cuboids[c]['dimensions']['length'], \
    #             cuboids[c]['dimensions']['width'], \
    #             cuboids[c]['dimensions']['height'], \
    #             rx = cuboids[c]['rotation']['x'], \
    #             ry = cuboids[c]['rotation']['y'], \
    #             rz = cuboids[c]['rotation']['z'], \
    #             rw = cuboids[c]['rotation']['w'])

    #         u = bbox.p2 - bbox.p1
    #         v = bbox.p4 - bbox.p1
    #         w = bbox.p5 - bbox.p1

    #         dot1 = np.dot(u[0:3], pc[i,:])
    #         #np.logical_and(dot1 >= np.dot(u,bboxx.p1[0:3]), dot1 <= np.dot(u,bboxx.p2[0:3]))

    #         dot2 = np.dot(v[0:3], pc[i,:])
    #         #np.logical_and(dot2 >= np.dot(v,bboxx.p1[0:3]), dot2 <= np.dot(v,bboxx.p4[0:3]))

    #         dot3 = np.dot(w[0:3], pc[i,:])
    #         #np.logical_and(dot3 >= np.dot(w,bboxx.p1[0:3]), dot2 <= np.dot(w,bboxx.p5[0:3]))

    #         if (in_between(dot1, np.dot(u[0:3],bbox.p1[0:3]),np.dot(u[0:3],bbox.p2[0:3]) ) and \
    #             in_between(dot2, np.dot(v[0:3],bbox.p1[0:3]),np.dot(v[0:3],bbox.p4[0:3]) ) and \
    #             in_between(dot3, np.dot(w[0:3],bbox.p1[0:3]),np.dot(w[0:3],bbox.p5[0:3]) ) ):
    #             import pdb; pdb.set_trace()
    #             pc_cuboid[c].append(pc[i,:])


def in_between(x, v1, v2, margin=0):
    return np.logical_or(
        np.logical_and(x >= v1 + margin, x <= v2 - margin),
        np.logical_and(x >= v2 + margin, x <= v1 - margin),
    )


def in_between_matrix(x, v1, v2):
    return np.logical_or(
        np.logical_and(x <= v1, x >= v2), np.logical_and(x <= v2, x >= v1)
    )


def write_txt(filename, s):
    f = open(filename, "w")
    f.write(s + "\n")
    f.close()


def get_pc_from_cuboid_world_only(
    pc_raw, cuboids, imgs, bboxs_crop, R_world, t_world, track_ids, crop_image=True
):
    pc_cuboids = []  # [None] * U.shape[0]
    bboxs = []  # [None] * U.shape[0]

    pc_cuboids_norm = []  # [None] * U.shape[0]
    bboxs_norm = []  # [None] * U.shape[0]
    R_norms = []  # [None] * U.shape[0]
    t_norms = []  # [None] * U.shape[0]
    pc_avgs = []  # [None] * U.shape[0]

    pc_cuboids_global = []  # [None] * U.shape[0]
    bboxs_global = []  # [None] * U.shape[0]
    R_norms_global = []  # [None] * U.shape[0]
    t_norms_global = []  # [None] * U.shape[0]
    pc_avgs_global = []  # [None] * U.shape[0]

    img_crops = []
    R_cameras = []
    t_cameras = []
    p_world = []
    p_global = []
    cuboids_select = []
    track_ids_select = []

    if len(pc_raw) < 100 or len(cuboids) == 0:
        return p_world, p_global, R_world, t_world
    U = []
    V = []
    W = []
    P1 = []
    P2 = []
    P4 = []
    P5 = []
    index = []
    for c in range(len(cuboids)):
        if not "center" in cuboids[c]:
            continue
        bbox = BBox3D(
            cuboids[c]["center"]["x"],
            cuboids[c]["center"]["y"],
            cuboids[c]["center"]["z"],
            cuboids[c]["dimensions"]["length"],
            cuboids[c]["dimensions"]["width"],
            cuboids[c]["dimensions"]["height"],
            rx=cuboids[c]["rotation"]["x"],
            ry=cuboids[c]["rotation"]["y"],
            rz=cuboids[c]["rotation"]["z"],
            rw=cuboids[c]["rotation"]["w"],
        )

        u = bbox.p2 - bbox.p1
        v = bbox.p4 - bbox.p1
        w = bbox.p5 - bbox.p1

        U.append(u[0:3])
        V.append(v[0:3])
        W.append(w[0:3])
        P1.append(bbox.p1[0:3])
        P2.append(bbox.p2[0:3])
        P4.append(bbox.p4[0:3])
        P5.append(bbox.p5[0:3])
        index.append(c)

    if len(U) == 0:
        return p_world, p_global, R_world, t_world

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

    for c in range(U.shape[0]):
        p = pc_raw[flag[c, :]]
        # import pdb; pdb.set_trace()

        if np.isnan(p).any() or np.isinf(p).any() or len(p) < 50:
            # print('NAN or INF detected!! Or point cloud contains too few points')
            # import pdb; pdb.set_trace()
            continue

        # try:

        # except:
        #    import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        p_world.append(np.matmul(R_world, p.transpose(1, 0)).transpose(1, 0) + t_world)
        p_global.append(p)
        cuboids_select.append(cuboids[c])
        track_ids_select.append(track_ids[c])
    return p_world, p_global, R_world, t_world, cuboids_select, track_ids_select

    # pc_avg_global = p.sum(axis=0)/len(p)
    # p = p - pc_avg_global
    # R_norm_global, t_norm_global = get_pc_normalize_keep_z(p)
    # p_norm = np.matmul(R_norm_global, p.transpose(1,0))+t_norm_global
    #


#
# p0_norm = np.matmul(R_norm_global, (P1[c,:]-pc_avg_global)[:,np.newaxis])+t_norm_global
# p1_norm = np.matmul(R_norm_global, (P2[c,:]-pc_avg_global)[:,np.newaxis])+t_norm_global
# p2_norm = np.matmul(R_norm_global, (P4[c,:]-pc_avg_global)[:,np.newaxis])+t_norm_global
#
# if p0_norm[0]>0: #align car head to x-axis!!!
#    R_pi = np.array([[-1,0,0], [0, -1,0],[0,0,1]])
#    p_pi = np.matmul(R_pi, p.transpose(1,0))
#    R_norm_global, t_norm_global = get_pc_normalize_keep_z(p_pi.transpose(1,0))
#
#    R_norm_global = np.matmul(R_norm_global, R_pi)
#    p_norm = np.matmul(R_norm_global, p.transpose(1,0))+t_norm_global
#
#    p0_norm = np.matmul(R_norm_global, (P1[c,:]-pc_avg_global)[:,np.newaxis])+t_norm_global
#    p1_norm = np.matmul(R_norm_global, (P2[c,:]-pc_avg_global)[:,np.newaxis])+t_norm_global
#    p2_norm = np.matmul(R_norm_global, (P4[c,:]-pc_avg_global)[:,np.newaxis])+t_norm_global
#
# h = np.linalg.norm(P5[c,:] - P1[c,:])
# bbox_norm = [ p0_norm , p1_norm , p2_norm , h ]
#
##import pdb; pdb.set_trace()
# x_norm = p1_norm-p0_norm
# x_norm[2] = 0
# x_norm = x_norm/np.linalg.norm(x_norm)
# x_axis = np.array([1,0,0])[:,np.newaxis]
#
# vv = np.cross(x_axis[:,0],x_norm[:,0])
##ss = np.linalg.norm(vv)
# cc = np.dot(x_norm[:,0],x_axis[:,0])
#
# vx = np.array([[0, -vv[2], vv[1]],[ vv[2], 0 , -vv[0]],[ -vv[1], vv[0], 0]])
#
# if cc == -1:
#    R_norm = np.array([[-1,0,0], [0, -1,0],[0,0,1]])
# else:
#    R_norm = np.eye(3) + vx + np.dot(vx,vx)* 1/(1+cc)
#
#
# t_norm = compute_bbox_center(bbox_norm)
##import pdb;pdb.set_trace()
#
# R_norm_inv = R_norm.transpose(1,0)
# t_norm_inv = -np.matmul(R_norm.transpose(1,0), t_norm)
# pc = np.matmul(R_norm_inv, p_norm)+t_norm_inv
#
# pc_avg = pc.transpose(1,0).sum(axis=0)/len(pc[0])
##R_norm, t_norm2 = get_pc_normalize_keep_z(pc.transpose(1,0)-pc_avg)
#
# R_norm_inv = R_norm.transpose(1,0)
# t_norm_inv = -np.matmul(R_norm.transpose(1,0), t_norm)
#
# p0 = np.matmul(R_norm_inv, p0_norm)+t_norm_inv+pc_avg[:,np.newaxis]
# p1 = np.matmul(R_norm_inv, p1_norm)+t_norm_inv+pc_avg[:,np.newaxis]
# p2 = np.matmul(R_norm_inv, p2_norm)+t_norm_inv+pc_avg[:,np.newaxis]
#
#
##bbox = np.concatenate((p0_norm.transpose(1,0), p1_norm.transpose(1,0), p2_norm.transpose(1,0),np.array(h).reshape(1,1)),axis=1)
#
# bbox_global = [P1[c,:][:,np.newaxis], P2[c,:][:, np.newaxis], P4[c,:][:, np.newaxis], h]
# bbox = [p0, p1,p2,h]
##import pdb;pdb.set_trace()
#
# pc_cuboids.append(pc.transpose(1,0))
# bboxs.append(bbox)
#
# pc_cuboids_norm.append(p_norm.transpose(1,0))
# bboxs_norm.append(bbox_norm)
# R_norms.append(R_norm)
# t_norms.append(t_norm)
# pc_avgs.append(pc_avg)
#
# pc_cuboids_global.append(pc_raw[flag[c,:]])
# bboxs_global.append(bbox_global)
# R_norms_global.append(R_norm_global)
# t_norms_global.append(t_norm_global)
# pc_avgs_global.append(pc_avg_global)
#
#
## bboxs.append(bbox)
## R_norms.append(R_norm)
## t_norms.append(t_norm)
## pc_avgs.append(pc_avg)
## pc_cuboids.append(p_norm.transpose(1,0))
#
# if crop_image:
#    bbox_crop = bboxs_crop[index[c]]
#    height, width, channels = imgs[index[c]].shape
#
#    #import pdb; pdb.set_trace()
#    img_crops.append(imgs[index[c]][int(bbox_crop[1]*height):int(bbox_crop[3]*height), int(bbox_crop[0]*width):int(bbox_crop[2]*width)])
#
##import pdb; pdb.set_trace()
# p_c = - pc_avg_global
# p_c = np.matmul(R_norm_global, p_c[:,np.newaxis])+t_norm_global
#
# z_c = -p_c/np.linalg.norm(p_c)
# x_c = np.array([-z_c[1][0], z_c[0][0], 0])
# x_c = x_c/np.linalg.norm(x_c)
#
# x_c = x_c[:,np.newaxis]
#
# y_c =  np.cross(z_c[:,0], x_c[:,0])
# if y_c[2] >0:
#    y_c = -y_c
#    x_c = -x_c
#
#
# y_c = y_c [:,np.newaxis]
#
#
# R_camera,t_camera = get_camera_matrix(p_c, x_c, y_c, z_c)
# R_cameras.append(R_camera)
# t_cameras.append(t_camera)

#    import pdb; pdb.set_trace()
# return pc_cuboids, bboxs, \
#    pc_cuboids_norm, bboxs_norm, R_norms, t_norms, pc_avgs, \
#    pc_cuboids_global, bboxs_global, R_norms_global, t_norms_global, pc_avgs_global,\
#    img_crops, R_cameras, t_cameras, p_world
