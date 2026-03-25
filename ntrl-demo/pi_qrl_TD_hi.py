from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import MLP, GCActor, GCDiscreteActor, LogParam, LengthNormalize, Identity
from utils.networks import GCIQEValue, GCMRNValue, GCMValue, GCLinfValue

class PI_QRL_TD_HI_Agent(flax.struct.PyTreeNode):
    """Hierarchical Quasimetric RL (QRL) agent.

    This implementation supports the following variants:
    (1) Value parameterizations: IQE (quasimetric_type='iqe') and MRN (quasimetric_type='mrn').
    (2) Actor losses: follows HiQL to improve generalization capabilities

    QRL with Hierarchical actor
    """
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)
    
    @jax.jit
    def distance_grad_s(self, obs, goals, grad_params):
        """Compute the quasimetric value function gradient penalty"""
        def distance_s(s, g, params):
            return self.network.select('value')(s, g, params=params).mean()
        
        grad_distance_s = jax.vmap(jax.grad(distance_s, argnums=0), in_axes=(0, 0, None))(obs, goals, grad_params)
        return grad_distance_s

    def value_loss(self, batch, grad_params):
        """Compute the QRL value loss."""
        d_sg = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)
        d_snext_g = self.network.select('value')(batch['next_observations'], batch['value_goals'], params=grad_params)
        grad_distance_sg = self.distance_grad_s(batch['observations'], batch['value_goals'], grad_params)

        if self.config['projection']:
            s_dot = batch['next_observations'] - batch['observations']
            grad_norm_d = jnp.sum(grad_distance_sg * s_dot, axis=-1)
        else:
            grad_norm_d = jnp.linalg.norm(grad_distance_sg + 1e-8, axis=-1) 

        eikonal_loss = jnp.mean((grad_norm_d - 1) ** 2) 

        total_loss = 0
        metrics = dict()
        
        q = -1 + self.config['discount']*batch['masks']*(-1)*d_snext_g
        q = jax.lax.stop_gradient(q)
        TD_loss = self.expectile_loss(q - (-1)*d_sg, q - (-1)*d_sg, self.config['expectile']).mean()

        total_loss += TD_loss 
        total_loss += eikonal_loss

        metrics['TD_loss'] = TD_loss
        metrics['Eikonal_loss'] = eikonal_loss

        metrics.update({'total_loss': total_loss,
            'd_sg_mean': d_sg.mean(),
            'd_sg_max': d_sg.max(),
            'd_sg_min': d_sg.min(),
            'grad_distance_sg_mean': grad_distance_sg.mean(),
            'grad_distance_sg_max': grad_distance_sg.max(),
            'grad_distance_sg_min': grad_distance_sg.min()
        })

        return total_loss, metrics

    def low_actor_loss(self, batch, grad_params):
        """Compute the low-level actor loss."""
        v = -self.network.select('value')(batch['observations'], batch['low_actor_goals'])
        nv = -self.network.select('value')(batch['next_observations'], batch['low_actor_goals'])
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        # Compute the goal representations of the subgoals.
        goal_reps = self.network.select('goal_rep')(jnp.concatenate([batch['observations'], 
                                                                     batch['low_actor_goals']], 
                                                                     axis=-1),
                                                                     params=grad_params)
        
        if not self.config['low_actor_rep_grad']:
            # Stop gradients through the goal representations.
            goal_reps = jax.lax.stop_gradient(goal_reps)

        dist = self.network.select('low_actor')(batch['observations'], 
                                                goal_reps, 
                                                goal_encoded=True, 
                                                params=grad_params)
        
        log_prob = dist.log_prob(batch['actions'])
        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }

        if not self.config['discrete']:
            actor_info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )

        return actor_loss, actor_info

    def high_actor_loss(self, batch, grad_params):
        """Compute the high-level actor loss."""
        v = -self.network.select('value')(batch['observations'], batch['high_actor_goals'])
        nv = -self.network.select('value')(batch['high_actor_targets'], batch['high_actor_goals'])
        adv = nv - v 

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('high_actor')(batch['observations'], 
                                                 batch['high_actor_goals'], 
                                                 params=grad_params)
        
        target = self.network.select('goal_rep')(jnp.concatenate([batch['observations'], 
                                                                  batch['high_actor_targets']], 
                                                                  axis=-1))
        
        log_prob = dist.log_prob(target)
        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - target) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        loss = value_loss + low_actor_loss + high_actor_loss
        return loss, info

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor.

        It first queries the high-level actor to obtain subgoal representations, and then queries the low-level actor
        to obtain raw actions.
        """
        high_seed, low_seed = jax.random.split(seed)

        high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)
        goal_reps = high_dist.sample(seed = high_seed)

        #normalization
        goal_reps = goal_reps / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True) * jnp.sqrt(goal_reps.shape[-1])

        low_dist = self.network.select('low_actor')(observations, goal_reps, goal_encoded=True, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)

        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        obs_dim = ex_observations.shape[-1]
        goal_dim = ex_observations.shape[-1] 

        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Length normalize encoder for the goal representation
        goal_rep_seq = []
        goal_rep_seq.append(LengthNormalize())
        goal_rep_def = nn.Sequential(goal_rep_seq)

        # Define (state-dependent) subgoal representation phi([s; g]) that outputs a length-normalized vector.
        # State-based environments only use the pre-defined shared encoder for subgoal representations.
        # Value: V(s, g)
        value_encoder_def = GCEncoder(state_encoder=Identity())

        # Low-level actor: pi^l(. | s, phi([s; w]))
        low_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)

        # High-level actor: pi^h(. | s, g) (i.e., no encoder)
        high_actor_encoder_def = None

        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=obs_dim + goal_dim, # config['rep_dim']
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=high_actor_encoder_def,
        )

        # Define value and actor networks.
        if config['quasimetric_type'] == 'mrn':
            value_def = GCMRNValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                layer_norm=config['layer_norm'],
                encoder=value_encoder_def,
            )

        elif config['quasimetric_type'] == 'iqe':
            value_def = GCIQEValue(
                hidden_dims=config['value_hidden_dims'],
                latent_dim=config['latent_dim'],
                dim_per_component=8,
                layer_norm=config['layer_norm'],
                encoder=value_encoder_def,
            )

        else:
            raise ValueError(f'Unsupported quasimetric type: {config["quasimetric_type"]}')

        if config['discrete']:
            low_actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=low_actor_encoder_def,
            )
        else:
            low_actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=low_actor_encoder_def,
            )

        network_info = dict(
            goal_rep=(goal_rep_def, (jnp.concatenate([ex_observations, ex_goals], axis=-1))),
            value=(value_def, (ex_observations, ex_goals)),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
            high_actor=(high_actor_def, (ex_observations, ex_goals)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='pi_qrl_TD_hi',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            quasimetric_type='iqe',  # Quasimetric parameterization type ('iqe' or 'mrn').
            latent_dim=512,  # Latent dimension for the quasimetric value function.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor (unused by default; can be used for geometric goal sampling in GCDataset).
            expectile=0.7,  # IQL expectile.
            low_alpha=3.0,  # Low-level AWR temperature.
            high_alpha=3.0,  # High-level AWR temperature.
            subgoal_steps=25,  # Subgoal steps.
            rep_dim=10,  # Goal representation dimension.
            low_actor_rep_grad=False,  # Whether low-actor gradients flow to goal representation (use True for pixels).
            const_std=True,  # Whether to use constant standard deviation for the actor.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            projection = False, # projecting gradient or not
            # Dataset hyperparameters.
            dataset_class='HGCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Unused (defined for compatibility with GCDataset).
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
