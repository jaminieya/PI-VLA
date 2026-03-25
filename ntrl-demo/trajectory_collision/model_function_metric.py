"""
Extended model_function_metric with collision-aware LossTrajectory.

Imports base Function from models.metric_arm and overrides LossTrajectory
to add collision avoidance loss when collision_callback is provided.
"""

from models.metric_arm import model_function_metric as _base
import torch


class Function(_base.Function):
    """Function with collision-aware LossTrajectory."""

    def LossTrajectory(self, points, tau_obs, beta, epoch, weight_traj=1.0, weight_eikonal=1e-2,
                      collision_callback=None, weight_collision=1e-3):
        """
        Trajectory loss with optional collision avoidance.
        - Trajectory supervision: (tau_pred - tau_obs)^2
        - Eikonal (F=1): (|nabla T|^2 - 1)^2
        - Collision: when collision_callback provided, adds normal-alignment loss
        """
        tau, w, Xp = self.network.out(points)
        dtau = self.gradient(tau, Xp)

        DT0 = dtau[:, :self.dim]
        DT1 = dtau[:, self.dim:]
        S0 = torch.einsum('ij,ij->i', DT0, DT0)
        S1 = torch.einsum('ij,ij->i', DT1, DT1)

        tau_pred = tau[:, 0]
        trajectory_loss = weight_traj * torch.mean((tau_pred - tau_obs) ** 2)
        eikonal_loss = weight_eikonal * (torch.mean((S0 - 1) ** 2) + torch.mean((S1 - 1) ** 2))
        loss_n = trajectory_loss + eikonal_loss

        if collision_callback is not None:
            Yobs, normal = collision_callback(points)
            normal0 = normal[:, :self.dim]
            normal1 = normal[:, self.dim:]
            n_loss0 = (1.01 - Yobs[:, 0].unsqueeze(1)) * (Yobs[:, 0].unsqueeze(1) * DT0 + normal0) ** 2
            n_loss1 = (1.01 - Yobs[:, 1].unsqueeze(1)) * (Yobs[:, 1].unsqueeze(1) * DT1 + normal1) ** 2
            collision_loss = weight_collision * (torch.sum(n_loss0, dim=1) + torch.sum(n_loss1, dim=1)).mean()
            loss_n = loss_n + collision_loss

        loss = beta * loss_n
        return loss, loss_n, trajectory_loss
