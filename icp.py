import pcl
import numpy as np


def run_icp(data):

    delta_theta_z, delta_x, delta_y, pc_in, pc_out, iter_t, iter_x, iter_y = data
    # if do_exhaustive_serach:
    transf_ini = np.eye(4)
    transf_ini[0, 0] = np.cos(delta_theta_z[iter_t])
    transf_ini[0, 1] = -np.sin(delta_theta_z[iter_t])
    transf_ini[1, 0] = np.sin(delta_theta_z[iter_t])
    transf_ini[1, 1] = np.cos(delta_theta_z[iter_t])
    transf_ini[0, 3] = delta_x[iter_x]
    transf_ini[1, 3] = delta_y[iter_y]

    pc_in_try = (
        np.matmul(transf_ini[0:3, 0:3], pc_in.transpose())
        + transf_ini[0:3, 3][:, np.newaxis]
    )
    pc_in_try = pc_in_try.transpose()

    cloud_in = pcl.PointCloud()
    cloud_out = pcl.PointCloud()

    cloud_in.from_array(pc_in_try.astype(np.float32))
    cloud_out.from_array(pc_out.astype(np.float32))

    gicp = cloud_in.make_GeneralizedIterativeClosestPoint()

    # tried open3d but found pcl version is more robust
    converged, transf_iter, estimate, fitness = gicp.gicp(
        cloud_in, cloud_out, max_iter=1000
    )

    if not converged:
        fitness = 0

    transf = np.eye(4)
    transf[0:3, 0:3] = np.matmul(transf_iter[0:3, 0:3], transf_ini[0:3, 0:3])
    transf[0:3, 3] = (
        np.matmul(transf_iter[0:3, 0:3], transf_ini[0:3, 3]) + transf_iter[0:3, 3]
    )

    # import pdb; pdb.set_trace()

    # pc_in_vec = PointCloud()
    # pc_in_vec.points = Vector3dVector(pc_in)
    # pc_out_vec = PointCloud()
    # pc_out_vec.points = Vector3dVector(pc_out)

    # draw_registration_result(pc_in_vec, pc_out_vec, transf)

    return transf, fitness
