"""
Standalone arm obstacle distance for collision checking.

Contains FK and arm_obstacle_distance without torch_IK_UR5 dependency.
Uses functorch for Jacobian computation.
"""

import math
import torch
from functorch import vmap, jacfwd


def _fk_and_jacobian(th_batch, chain, mesh_list):
    """Compute FK positions and Jacobian for arm sphere points."""
    def chain_to_matrix(th_batch):
        tg_batch = chain.forward_kinematics(th_batch, end_only=False)
        m_list = []
        iter_count = 0
        for tg in tg_batch:
            if 1 < iter_count < 8:
                m = tg_batch[tg].get_matrix()
                m_list.append(m)
            iter_count += 1
        return torch.cat(m_list, dim=0)

    def output_and_jacobian_fn(th_batch):
        output = chain_to_matrix(th_batch)
        jacobian = jacfwd(chain_to_matrix)(th_batch)
        return output, jacobian

    matrix_list, jacobian = vmap(output_and_jacobian_fn)(th_batch)
    matrix_list = matrix_list.detach()
    jacobian = jacobian.detach()

    p_list = []
    gradient_p_list = []
    for iter_idx in range(6):
        ball_list = mesh_list[iter_idx]
        ones_column = torch.ones(ball_list.size(0), 1, device=ball_list.device)
        nv = torch.cat((ball_list[:, :3], ones_column), dim=1)
        m = matrix_list[:, iter_idx, ...]
        gradient_m = jacobian[:, iter_idx, ...].permute(0, 3, 1, 2)
        p = torch.matmul(m, nv.T)
        p = torch.permute(p, (0, 2, 1))
        p[..., 3] = ball_list[:, 3]
        p_list.append(p)
        gradient_p = torch.matmul(gradient_m, nv.T)
        gradient_p_list.append(gradient_p)

    return p_list, gradient_p_list


def arm_obstacle_distance(th_batch, chain, mesh_list, kdtree, v_obs):
    """
    Compute minimum obstacle distance and its gradient w.r.t. joint configs.

    Args:
        th_batch: (N, 6) joint configs in radians on CUDA.
        chain: UR5 kinematic chain.
        mesh_list: Sphere mesh list.
        kdtree: Obstacle KD-tree.
        v_obs: Obstacle point cloud.

    Returns:
        (distance, normal): distance (N,), normal (N, 6).
    """
    device = th_batch.device
    batch_size_limit = 200000
    n_total = th_batch.shape[0]
    whole_dis = torch.zeros((n_total,), dtype=torch.float32, device=device)
    whole_normal = torch.zeros((n_total, 6), dtype=torch.float32, device=device)

    for start in range(0, n_total, batch_size_limit):
        end = min(start + batch_size_limit, n_total)
        chunk = th_batch[start:end]
        curr_batch = chunk.shape[0]

        p_list, gradient_p_list = _fk_and_jacobian(chunk, chain, mesh_list)

        query_points = torch.cat(p_list, dim=1)
        query_points_grad = torch.cat(gradient_p_list, dim=3)
        del p_list, gradient_p_list

        query_points_grad = query_points_grad.permute(0, 3, 1, 2)
        query_points = torch.reshape(query_points, (-1, 4))
        query_points_grad = torch.reshape(query_points_grad, (-1, 6, 4))

        dists, inds = kdtree.query(query_points[:, :3], nr_nns_searches=1)
        dists = dists.squeeze()
        inds = inds.squeeze()
        distance = torch.sqrt(dists) - query_points[:, 3]

        normal = query_points[:, :3] - v_obs[inds, :]
        del inds, query_points
        normal = torch.bmm(query_points_grad[..., :3], normal.unsqueeze(2)).squeeze(2)
        del query_points_grad

        distance = distance.reshape(curr_batch, -1)
        normal = normal.reshape(curr_batch, -1, normal.shape[-1])

        arg_min = torch.argmin(distance, dim=1, keepdim=True)
        min_distance = torch.gather(distance, 1, arg_min).squeeze(1)
        min_normal = torch.gather(normal, 1, (arg_min.unsqueeze(2)).expand(-1, -1, 6)).squeeze(1)
        del distance, normal, arg_min

        whole_dis[start:end] = min_distance
        whole_normal[start:end, :] = min_normal
        del min_distance, min_normal

        torch.cuda.empty_cache()

    return whole_dis, whole_normal
