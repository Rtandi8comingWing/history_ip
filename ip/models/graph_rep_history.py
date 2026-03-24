from torch_geometric.data import HeteroData
from torch import nn
import torch
import numpy as np
from ip.utils.common_utils import printarr, PositionalEncoder
from ip.utils.common_utils import SinusoidalPosEmb


class GraphRep(nn.Module):
    def __init__(self, config):
        super(GraphRep, self).__init__()
        ################################################################################################################
        # Store the parameters.
        self.batch_size = config['batch_size']
        self.num_demos = config['num_demos']
        self.traj_horizon = config['traj_horizon']
        self.num_scenes_nodes = config['num_scenes_nodes']
        self.num_freqs = config['local_num_freq']
        self.device = config['device']
        self.embd_dim = config['local_nn_dim']
        self.pred_horizon = config['pre_horizon']
        self.pos_in_nodes = config['pos_in_nodes']

        # =========================================================================
        # HA-IGD: Track node configuration
        # =========================================================================
        self.enable_track_nodes = config.get('enable_track_nodes', False)
        self.track_n_max = config.get('track_n_max', 5)
        self.track_history_len = config.get('track_history_len', 16)
        self.track_points_per_obj = config.get('track_points_per_obj', 5)
        self.track_hidden_dim = config.get('track_hidden_dim', 512)
        self.track_age_embed_dim = config.get('track_age_embed_dim', 32)
        self.track_age_norm_max_sec = config.get('track_age_norm_max_sec', 2.0)

        # Soft membership
        self.soft_membership_sigma = config.get('soft_membership_sigma', 0.05)

        # Graph topology flags (V0)
        self.enable_track_track_edges = config.get('enable_track_track_edges', False)
        self.enable_track_geo_edges = config.get('enable_track_geo_edges', True)
        self.enable_demo_current_track_edges = config.get('enable_demo_current_track_edges', True)
        self.enable_current_track_to_action_edges = config.get('enable_current_track_to_action_edges', True)

        # Curriculum dropout
        self.curriculum_dropout_rate = config.get('curriculum_dropout_start', 0.05)
        self.track_modality_dropout_eval = config.get('track_modality_dropout_eval', 0.0)

        # Track encoder (imported lazily to avoid circular imports)
        self._track_encoder = None

        # =========================================================================
        # HA-IGD: Track node embeddings (learnable)
        # =========================================================================
        if self.enable_track_nodes:
            self.track_embds = nn.Embedding(
                self.track_n_max,
                self.embd_dim, device=self.device)
        ################################################################################################################
        # These will be used to represent the gripper node positions.
        self.gripper_node_pos = torch.tensor([
            [0., 0., 0.],  # Middle of the gripper.
            [0., 0., -0.03],  # Tail of the gripper.
            [0., 0.03, 0.],  # Side of the gripper.
            [0., -0.03, 0.],  # Side of the gripper.
            [0., 0.03, 0.03],  # Finger.
            [0., -0.03, 0.03],  # finger.
        ], dtype=torch.float32, device=self.device) * 2
        self.num_g_nodes = len(self.gripper_node_pos)
        self.g_state_dim = 64
        self.d_time_dim = 64
        ################################################################################################################
        self.sine_pos_embd = SinusoidalPosEmb(self.d_time_dim)  # Think about this
        # Define the learnable embeddings for the nodes and edges.
        self.pos_embd = PositionalEncoder(3, self.num_freqs, log_space=True, add_original_x=True, scale=1.0)
        self.edge_dim = self.pos_embd.d_output * 2
        self.gripper_proj = nn.Linear(1, self.g_state_dim)

        self.gripper_embds = nn.Embedding(
            len(self.gripper_node_pos) * (self.pred_horizon + 1),
            self.embd_dim - self.g_state_dim, device=self.device)
        self.gripper_cond_gripper_embds = nn.Embedding(1, self.edge_dim, device=self.device)
        self.gripper_da_gripper_embds = nn.Embedding(1, self.edge_dim, device=self.device)
        ################################################################################################################
        # Define the structure of the graph.
        self.node_types = ['scene', 'gripper', 'track']
        self.edge_types = [
            # Local observation subgraphs.
            ('scene', 'rel', 'scene'),
            ('scene', 'rel', 'gripper'),
            ('gripper', 'rel', 'gripper'),
            # Propagation from demo gripper to current gripper.
            ('gripper', 'cond', 'gripper'),
            # Propagating information about the timestep in the demo.
            ('gripper', 'demo', 'gripper'),
            # Propagating information about the timestep in the demo.
            ('gripper', 'time_action', 'gripper'),
            # Propagating information from demo to action.
            ('gripper', 'demo_action', 'gripper'),
            # HA-IGD: Track edges
            ('track', 'rel', 'track'),
            ('track', 'rel', 'scene'),
            ('track', 'rel', 'gripper'),
        ]
        self.graph = None
        ################################################################################################################

    def create_dense_edge_idx(self, num_nodes_source, num_nodes_dest):
        return torch.cartesian_prod(
            torch.arange(num_nodes_source, dtype=torch.int64, device=self.device),
            torch.arange(num_nodes_dest, dtype=torch.int64, device=self.device)).contiguous().t()

    def get_node_info(self):
        # A bunch of arange operations to store information which node in the graph belongs to which batch, timestep etc.
        # First the scene nodes. [bs, nd, th, sn, 3] + [bs, sn, 3]
        ################################################################################################################
        sb = torch.arange(self.batch_size, device=self.device)
        scene_batch = sb[:, None, None, None].repeat(1,
                                                     self.num_demos,
                                                     self.traj_horizon,
                                                     self.num_scenes_nodes
                                                     ).view(-1)
        sb_current = sb[:, None].repeat(1, self.num_scenes_nodes).view(-1)
        scene_batch = torch.cat([scene_batch, sb_current], dim=0)

        scene_traj = torch.arange(self.traj_horizon,
                                  device=self.device)[None, None, :, None].repeat(self.batch_size,
                                                                                  self.num_demos,
                                                                                  1,
                                                                                  self.num_scenes_nodes
                                                                                  ).view(-1)
        scene_traj = torch.cat([scene_traj, self.traj_horizon * torch.ones_like(sb_current)], dim=0)

        scene_demo = torch.arange(self.num_demos, device=self.device)[None, :, None, None].repeat(
            self.batch_size, 1, self.traj_horizon, self.num_scenes_nodes).view(-1)
        scene_current = self.num_demos * torch.ones(self.batch_size * self.num_scenes_nodes, device=self.device)
        scene_demo = torch.cat([scene_demo, scene_current], dim=0)

        # Accounting for scene action nodes.
        scene_batch_action = sb[:, None, None].repeat(1, self.pred_horizon, self.num_scenes_nodes).view(-1)
        scene_batch = torch.cat([scene_batch, scene_batch_action], dim=0)

        scene_traj_action = torch.arange(self.pred_horizon, device=self.device)[None, :, None].repeat(
            self.batch_size, 1, self.num_scenes_nodes).view(-1) + self.traj_horizon + 1
        scene_traj = torch.cat([scene_traj, scene_traj_action], dim=0)
        scene_demo = torch.cat([scene_demo, self.num_demos * torch.ones_like(scene_traj_action)], dim=0)
        ################################################################################################################
        # Now the gripper nodes. [bs, nd, th, gn, 3] + [bs, gn, 3] + [bs, ph, gn, 3]
        gripper_batch = sb[:, None, None, None].repeat(1, self.num_demos, self.traj_horizon, self.num_g_nodes).view(-1)
        gripper_batch_current = sb[:, None].repeat(1, self.num_g_nodes).view(-1)
        gripper_batch_action = sb[:, None, None].repeat(1, self.pred_horizon, self.num_g_nodes).view(-1)
        gripper_batch = torch.cat([gripper_batch, gripper_batch_current, gripper_batch_action], dim=0)

        gripper_time = torch.arange(self.traj_horizon, device=self.device, dtype=torch.long)[None, None, :,
                       None].repeat(self.batch_size, self.num_demos, 1, self.num_g_nodes).view(-1)

        gripper_time_current = self.traj_horizon * torch.ones(self.batch_size * self.num_g_nodes, device=self.device,
                                                              dtype=torch.long)
        gripper_time_action = torch.arange(self.pred_horizon, device=self.device, dtype=torch.long)[None, :,
                              None].repeat(self.batch_size, 1, self.num_g_nodes).view(-1)
        gripper_time = torch.cat([gripper_time,
                                  gripper_time_current,
                                  gripper_time_action + self.traj_horizon + 1], dim=0)

        gripper_node = torch.arange(self.num_g_nodes, device=self.device)[None, None, None, :].repeat(
            self.batch_size, self.num_demos, self.traj_horizon, 1).view(-1)
        gripper_node_current = torch.arange(self.num_g_nodes, device=self.device)[None, :].repeat(
            self.batch_size, 1).view(-1)
        gripper_node_action = torch.arange(self.num_g_nodes, device=self.device)[None, None, :].repeat(
            self.batch_size, self.pred_horizon, 1).view(-1)
        gripper_node = torch.cat([gripper_node, gripper_node_current, gripper_node_action], dim=0)

        gripper_emdb = gripper_node
        gripper_emdb[gripper_time > self.traj_horizon] += self.num_g_nodes * gripper_time_action

        gripper_demo = torch.arange(self.num_demos, device=self.device)[None, :, None, None].repeat(
            self.batch_size, 1, self.traj_horizon, self.num_g_nodes).view(-1)
        gripper_current = self.num_demos * torch.ones(self.batch_size * (self.pred_horizon + 1) * self.num_g_nodes,
                                                      device=self.device)
        gripper_demo = torch.cat([gripper_demo, gripper_current], dim=0)

        # =========================================================================
        # HA-IGD: Track nodes (per batch, per track index)
        # Only for current observation (not demos), max track_n_max tracks per batch
        # =========================================================================
        if self.enable_track_nodes:
            track_batch = sb[:, None].repeat(1, self.track_n_max).view(-1)  # [B * Nmax]
            track_idx = torch.arange(self.track_n_max, device=self.device)[None, :].repeat(
                self.batch_size, 1).view(-1)  # [B * Nmax]
        else:
            track_batch = torch.tensor([], dtype=torch.long, device=self.device)
            track_idx = torch.tensor([], dtype=torch.long, device=self.device)

        return {
            'scene': {
                'batch': scene_batch,
                'traj': scene_traj,
                'demo': scene_demo,
            },
            'gripper': {
                'batch': gripper_batch,
                'time': gripper_time,
                'node': gripper_node,
                'embd': gripper_emdb,
                'demo': gripper_demo,
            },
            'track': {
                'batch': track_batch,
                'track_idx': track_idx,
            }
        }

    def transform_gripper_nodes(self, gripper_nodes, T):
        # gripper_nodes - [B, D, T, N, 3]
        # T - [B, D, T, 4, 4]
        has_demo = len(gripper_nodes.shape) == 5
        if not has_demo:
            gripper_nodes = gripper_nodes.unsqueeze(1)
        b, d, t, n, _ = gripper_nodes.shape
        gripper_nodes = gripper_nodes.reshape(-1, gripper_nodes.shape[-2], gripper_nodes.shape[-1]).permute(0, 2, 1)
        gripper_nodes = torch.bmm(T[..., :3, :3].reshape(-1, 3, 3), gripper_nodes)
        gripper_nodes += T[..., :3, 3].reshape(-1, 3, 1)
        gripper_nodes = gripper_nodes.permute(0, 2, 1).view(b, d, t, n, 3)
        if not has_demo:
            gripper_nodes = gripper_nodes.squeeze(1)
        return gripper_nodes

    def initialise_graph(self):
        # Manually connecting different nodes in the graph to achieve our desired graph representation.
        # Probably could be re-written to be more beautiful. Most definitely could.
        self.graph = HeteroData()
        node_info = self.get_node_info()

        dense_g_g = self.create_dense_edge_idx(node_info['gripper']['embd'].shape[0],
                                               node_info['gripper']['embd'].shape[0])

        dense_s_s = self.create_dense_edge_idx(node_info['scene']['batch'].shape[0],
                                               node_info['scene']['batch'].shape[0])

        dense_s_g = self.create_dense_edge_idx(node_info['scene']['batch'].shape[0],
                                               node_info['gripper']['embd'].shape[0])
        ################################################################################################################
        s_rel_s_mask = node_info['scene']['batch'][dense_s_s[0, :]] == node_info['scene']['batch'][dense_s_s[1, :]]
        s_rel_s_mask = s_rel_s_mask & (
                node_info['scene']['traj'][dense_s_s[0, :]] == node_info['scene']['traj'][dense_s_s[1, :]])
        s_rel_s_mask = s_rel_s_mask & (
                node_info['scene']['demo'][dense_s_s[0, :]] == node_info['scene']['demo'][dense_s_s[1, :]])
        ################################################################################################################
        s_rel_s_action_mask = s_rel_s_mask & (
                node_info['scene']['traj'][dense_s_s[0, :]] > self.traj_horizon)
        s_rel_s_action_mask = s_rel_s_action_mask & (
                node_info['scene']['traj'][dense_s_s[1, :]] > self.traj_horizon)
        s_rel_s_mask_demo = s_rel_s_mask & torch.logical_not(s_rel_s_action_mask)
        ################################################################################################################
        g_rel_g_mask = node_info['gripper']['batch'][dense_g_g[0, :]] == node_info['gripper']['batch'][dense_g_g[1, :]]
        g_rel_g_mask = g_rel_g_mask & (
                node_info['gripper']['time'][dense_g_g[0, :]] == node_info['gripper']['time'][dense_g_g[1, :]])
        g_rel_g_mask = g_rel_g_mask & (
                node_info['gripper']['demo'][dense_g_g[0, :]] == node_info['gripper']['demo'][dense_g_g[1, :]])
        ################################################################################################################
        s_rel_g_mask = node_info['scene']['batch'][dense_s_g[0, :]] == node_info['gripper']['batch'][dense_s_g[1, :]]
        s_rel_g_mask = s_rel_g_mask & (
                node_info['scene']['traj'][dense_s_g[0, :]] == node_info['gripper']['time'][dense_s_g[1, :]])
        s_rel_g_mask = s_rel_g_mask & (
                node_info['scene']['demo'][dense_s_g[0, :]] == node_info['gripper']['demo'][dense_s_g[1, :]])
        ################################################################################################################
        s_rel_g_action_mask = s_rel_g_mask & (
                node_info['scene']['traj'][dense_s_g[0, :]] > self.traj_horizon)
        s_rel_g_action_mask = s_rel_g_action_mask & (
                node_info['gripper']['time'][dense_s_g[1, :]] > self.traj_horizon)
        s_rel_g_mask_demo = s_rel_g_mask & torch.logical_not(s_rel_g_action_mask)
        ################################################################################################################
        g_c_g_mask = node_info['gripper']['batch'][dense_g_g[0, :]] == node_info['gripper']['batch'][dense_g_g[1, :]]
        g_c_g_mask = g_c_g_mask & (
                node_info['gripper']['time'][dense_g_g[0, :]] < self.traj_horizon)
        g_c_g_mask = g_c_g_mask & (node_info['gripper']['time'][dense_g_g[1, :]] == self.traj_horizon)
        ################################################################################################################
        g_t_g_mask = node_info['gripper']['batch'][dense_g_g[0, :]] == node_info['gripper']['batch'][dense_g_g[1, :]]
        g_t_g_mask = g_t_g_mask & (node_info['gripper']['time'][dense_g_g[0, :]] >= self.traj_horizon)
        g_t_g_mask = g_t_g_mask & (node_info['gripper']['time'][dense_g_g[1, :]] > self.traj_horizon)
        g_t_g_mask = g_t_g_mask & (
                node_info['gripper']['time'][dense_g_g[1, :]] != node_info['gripper']['time'][dense_g_g[0, :]])
        g_tc_g = g_t_g_mask & (node_info['gripper']['time'][dense_g_g[0, :]] == self.traj_horizon)
        g_t_g_mask = g_t_g_mask & torch.logical_not(g_tc_g)
        ################################################################################################################
        g_d_g_mask = node_info['gripper']['batch'][dense_g_g[0, :]] == node_info['gripper']['batch'][dense_g_g[1, :]]
        g_d_g_mask = g_d_g_mask & (node_info['gripper']['time'][dense_g_g[0, :]] < self.traj_horizon)
        g_d_g_mask = g_d_g_mask & (node_info['gripper']['time'][dense_g_g[1, :]] < self.traj_horizon)
        g_d_g_mask = g_d_g_mask & (
                node_info['gripper']['time'][dense_g_g[0, :]] != node_info['gripper']['time'][dense_g_g[1, :]])
        g_d_g_mask = g_d_g_mask & (
                node_info['gripper']['demo'][dense_g_g[0, :]] == node_info['gripper']['demo'][dense_g_g[1, :]])
        g_d_g_mask = g_d_g_mask & (node_info['gripper']['time'][dense_g_g[1, :]] - node_info['gripper']['time'][
            dense_g_g[0, :]] == -1)
        ################################################################################################################
        self.graph.gripper_batch = node_info['gripper']['batch']
        self.graph.gripper_time = node_info['gripper']['time']
        self.graph.gripper_node = node_info['gripper']['node']
        self.graph.gripper_embd = node_info['gripper']['embd'].long()
        self.graph.gripper_demo = node_info['gripper']['demo']
        self.graph.scene_batch = node_info['scene']['batch']
        self.graph.scene_traj = node_info['scene']['traj']
        self.graph.scene_demo = node_info['scene']['demo']

        self.graph[('gripper', 'rel', 'gripper')].edge_index = dense_g_g[:, g_rel_g_mask]
        self.graph[('scene', 'rel', 'scene')].edge_index = dense_s_s[:, s_rel_s_mask]
        self.graph[('scene', 'rel', 'gripper')].edge_index = dense_s_g[:, s_rel_g_mask]
        self.graph[('gripper', 'cond', 'gripper')].edge_index = dense_g_g[:, g_c_g_mask]
        self.graph[('gripper', 'time_action', 'gripper')].edge_index = dense_g_g[:, g_t_g_mask]
        self.graph[('gripper', 'demo', 'gripper')].edge_index = dense_g_g[:, g_d_g_mask]

        self.graph[('scene', 'rel_action', 'gripper')].edge_index = dense_s_g[:, s_rel_g_action_mask]
        self.graph[('scene', 'rel_demo', 'gripper')].edge_index = dense_s_g[:, s_rel_g_mask_demo]
        self.graph[('scene', 'rel_action', 'scene')].edge_index = dense_s_s[:, s_rel_s_action_mask]
        self.graph[('scene', 'rel_demo', 'scene')].edge_index = dense_s_s[:, s_rel_s_mask_demo]
        self.graph[('gripper', 'rel_cond', 'gripper')].edge_index = dense_g_g[:, g_tc_g]

        # =========================================================================
        # HA-IGD: Track node edges
        # =========================================================================
        if self.enable_track_nodes and 'track' in node_info and node_info['track']['batch'].shape[0] > 0:
            num_track_nodes = node_info['track']['batch'].shape[0]

            # Dense edges for track-track, track-scene, track-gripper
            dense_t_t = self.create_dense_edge_idx(num_track_nodes, num_track_nodes)

            # Current scene nodes are at index after demo scenes: (demo * traj + current)
            # scene nodes: demo scenes + current scene + action scenes
            # current scene starts at: batch_size * num_demos * traj_horizon * num_scenes_nodes
            num_scene_nodes = node_info['scene']['batch'].shape[0]
            num_gripper_nodes = node_info['gripper']['batch'].shape[0]

            # For track->scene: connect each track node to current scene nodes (not demo)
            # Current scene nodes are at indices: [demo_scenes : demo_scenes + current_scenes]
            demo_scene_count = self.batch_size * self.num_demos * self.traj_horizon * self.num_scenes_nodes
            current_scene_start = demo_scene_count
            current_scene_count = self.batch_size * self.num_scenes_nodes
            current_scene_end = current_scene_start + current_scene_count

            # Track to current scene
            dense_t_s = torch.cartesian_prod(
                torch.arange(num_track_nodes, dtype=torch.long, device=self.device),
                torch.arange(current_scene_start, current_scene_end, dtype=torch.long, device=self.device)
            ).t()

            # Track to gripper (current gripper at traj_horizon)
            # Gripper nodes: demo gripper + current gripper + action gripper
            demo_gripper_count = self.batch_size * self.num_demos * self.traj_horizon * self.num_g_nodes
            current_gripper_start = demo_gripper_count
            current_gripper_count = self.batch_size * self.num_g_nodes

            dense_t_g = torch.cartesian_prod(
                torch.arange(num_track_nodes, dtype=torch.long, device=self.device),
                torch.arange(current_gripper_start, current_gripper_start + current_gripper_count, dtype=torch.long, device=self.device)
            ).t()

            # Edge masks
            # Track-track: same batch
            t_rel_t_mask = node_info['track']['batch'][dense_t_t[0]] == node_info['track']['batch'][dense_t_t[1]]

            # Track-scene: same batch
            # Scene batch indices: scene_batch values for current scene nodes
            current_scene_batch = node_info['scene']['batch'][current_scene_start:current_scene_end]
            t_rel_s_mask = node_info['track']['batch'][dense_t_s[0]] == current_scene_batch[dense_t_s[1] - current_scene_start]

            # Track-gripper: same batch
            current_gripper_batch = node_info['gripper']['batch'][current_gripper_start:current_gripper_start + current_gripper_count]
            t_rel_g_mask = node_info['track']['batch'][dense_t_g[0]] == current_gripper_batch[dense_t_g[1] - current_gripper_start]

            # Assign edge indices
            self.graph[('track', 'rel', 'track')].edge_index = dense_t_t[:, t_rel_t_mask]
            self.graph[('track', 'rel', 'scene')].edge_index = dense_t_s[:, t_rel_s_mask]
            self.graph[('track', 'rel', 'gripper')].edge_index = dense_t_g[:, t_rel_g_mask]

            # Store track node metadata
            self.graph.track_batch = node_info['track']['batch']
            self.graph.track_idx = node_info['track']['track_idx']

    def update_graph(self, data):
        # Adding information to the graph structure create in initialise_graph.
        # scene_node_pos: # [B, N, T, S, 3]
        gripper_node_pos = self.gripper_node_pos[None, None, None, :, :].repeat(self.batch_size,
                                                                                self.num_demos,
                                                                                self.traj_horizon, 1, 1)
        ################################################################################################################
        # demo_T_w_es: [B, D, T, 4, 4]
        # T_w_e: [B, 4, 4]
        # T_w_n: [B, P, 4, 4]
        # Create identity matrix like T_w_e
        I_w_e = torch.eye(4, device=self.device)[None, :, :].repeat(self.batch_size, 1, 1)

        all_T_w_e = torch.cat([
            data.demo_T_w_es[:, :self.num_demos, :, None, :, :].repeat(1, 1, 1, 6, 1, 1).view(-1, 4, 4),
            I_w_e[:, None, :, :].repeat(1, 6, 1, 1).view(-1, 4, 4),
            data.actions[:, :, None, :, :].repeat(1, 1, 6, 1, 1).view(-1, 4, 4)
        ])
        all_T_e_w = all_T_w_e.inverse()
        ################################################################################################################

        gripper_node_pos_current = gripper_node_pos[:, 0, 0, ...].view(self.batch_size, -1, 3)
        gripper_node_pos_action = self.gripper_node_pos[None, None, :, :].repeat(self.batch_size,
                                                                                 self.pred_horizon, 1, 1)

        gripper_node_pos = torch.cat([gripper_node_pos.reshape(-1, 3),
                                      gripper_node_pos_current.reshape(-1, 3),
                                      gripper_node_pos_action.reshape(-1, 3)], dim=0)

        # data.graps_demos [B, D, T, 1]
        gripper_states = self.gripper_proj(data.graps_demos[:, :self.num_demos])[..., None, :].repeat(1, 1, 1,
                                                                                                      self.num_g_nodes,
                                                                                                      1)
        gripper_states = gripper_states.view(-1, self.g_state_dim)
        gripper_states_current = self.gripper_proj(data.current_grip.unsqueeze(-1))[..., None, :].repeat(1,
                                                                                                         self.num_g_nodes,
                                                                                                         1)
        gripper_states_current = gripper_states_current.view(-1, self.g_state_dim)
        gripper_states_action = self.gripper_proj(data.actions_grip.unsqueeze(-1))[..., None, :].repeat(1, 1,
                                                                                                        self.num_g_nodes,
                                                                                                        1)
        gripper_states_action = gripper_states_action.view(-1, self.g_state_dim)
        gripper_states = torch.cat([gripper_states, gripper_states_current, gripper_states_action], dim=0)
        gripper_embd = self.gripper_embds(self.graph.gripper_embd)

        # Adding diffusion time step information to gripper action nodes.
        d_time_embd = self.sine_pos_embd(data.diff_time)[:, None, ...].repeat(1,
                                                                              self.pred_horizon,
                                                                              self.num_g_nodes,
                                                                              1).view(-1, self.d_time_dim)
        gripper_embd[self.graph.gripper_time > self.traj_horizon][:, -self.d_time_dim:] = d_time_embd

        gripper_embd = torch.cat([gripper_embd, gripper_states], dim=-1)

        scene_node_pos = torch.cat([
            data.demo_scene_node_pos[:, :self.num_demos].reshape(-1, 3),
            data.live_scene_node_pos.view(-1, 3),
            data.action_scene_node_pos.view(-1, 3)
        ], dim=0)
        scene_node_embd = torch.cat([
            data.demo_scene_node_embds[:, :self.num_demos].reshape(-1, self.embd_dim),
            data.live_scene_node_embds.view(-1, self.embd_dim),
            data.action_scene_node_embds.view(-1, self.embd_dim)
        ], dim=0)

        self.graph['gripper'].pos = gripper_node_pos
        self.graph['gripper'].x = gripper_embd
        self.graph['scene'].pos = scene_node_pos
        self.graph['scene'].x = scene_node_embd

        # =========================================================================
        # HA-IGD: Track node position and features
        # =========================================================================
        if self.enable_track_nodes and hasattr(data, 'track_node_embds') and data.track_node_embds is not None:
            # data.track_node_embds: [B, N, embd_dim]
            B, N, Emb = data.track_node_embds.shape
            track_emb_flat = data.track_node_embds.reshape(-1, Emb)  # [B*N, embd_dim]

            # Track positions: use last point of each track as position
            # data.current_track_seq: [B, N, H, P, 3]
            if hasattr(data, 'current_track_seq') and data.current_track_seq is not None:
                # Take last timestep, last point of each track
                track_pos = data.current_track_seq[:, :, -1, 0, :]  # [B, N, 3] - use first point per object for simplicity
                # Actually take mean of last frame
                track_pos = data.current_track_seq[:, :, -1, :, :].mean(dim=2)  # [B, N, 3]
                track_pos_flat = track_pos.reshape(-1, 3)  # [B*N, 3]
            else:
                # Fallback: zero positions
                track_pos_flat = torch.zeros(B * N, 3, device=self.device)

            # Assign to graph
            self.graph['track'].pos = track_pos_flat
            self.graph['track'].x = track_emb_flat

            # Add positional encoding if enabled
            if self.pos_in_nodes:
                self.graph['track'].x = torch.cat(
                    [self.graph['track'].x, self.pos_embd(self.graph['track'].pos)], dim=-1)
        else:
            # No track data - create empty placeholder
            if self.enable_track_nodes:
                # Create empty track node
                self.graph['track'].pos = torch.empty(0, 3, device=self.device)
                self.graph['track'].x = torch.empty(0, self.embd_dim, device=self.device)

        if self.pos_in_nodes:
            self.graph['gripper'].x = \
                torch.cat([self.graph['gripper'].x, self.pos_embd(self.graph['gripper'].pos)], dim=-1)
            self.graph['scene'].x = \
                torch.cat([self.graph['scene'].x, self.pos_embd(self.graph['scene'].pos)], dim=-1)

        self.add_rel_edge_attr('scene', 'gripper')
        self.add_rel_edge_attr('gripper', 'gripper')
        self.add_rel_edge_attr('scene', 'scene')

        self.graph[('gripper', 'cond', 'gripper')].edge_attr = self.gripper_cond_gripper_embds(
            torch.zeros(len(self.graph[('gripper', 'cond', 'gripper')].edge_index[0]), device=self.device).long())

        self.add_rel_edge_attr('scene', 'gripper', edge='rel_action')
        self.add_rel_edge_attr('scene', 'gripper', edge='rel_demo')

        self.add_rel_edge_attr('scene', 'scene', edge='rel_demo')
        self.add_rel_edge_attr('scene', 'scene', edge='rel_action')

        self.add_rel_edge_attr('gripper', 'gripper', edge='time_action',
                               all_T_w_e=all_T_w_e, all_T_e_w=all_T_e_w)
        self.add_rel_edge_attr('gripper', 'gripper', edge='rel_cond',
                               all_T_w_e=all_T_w_e, all_T_e_w=all_T_e_w)
        self.add_rel_edge_attr('gripper', 'gripper', edge='demo',
                               all_T_w_e=all_T_w_e, all_T_e_w=all_T_e_w)

        # =========================================================================
        # HA-IGD: Track edge attributes
        # =========================================================================
        if self.enable_track_nodes:
            self.add_rel_edge_attr('track', 'track')
            self.add_rel_edge_attr('track', 'scene')
            self.add_rel_edge_attr('track', 'gripper')

    def add_rel_edge_attr(self, source, dest, edge='rel', all_T_w_e=None, all_T_e_w=None):
        # Retrieve edge index for this relation
        edge_idx = self.graph[(source, edge, dest)].edge_index

        # If there are no edges of this type, create an empty edge_attr tensor and return early
        if edge_idx.numel() == 0:
            empty_attr = torch.empty((0, self.edge_dim), device=self.device)
            self.graph[(source, edge, dest)].edge_attr = empty_attr
            return

        # Validate and filter edge indices to prevent out-of-bounds access
        num_source_nodes = self.graph[source].pos.shape[0]
        num_dest_nodes = self.graph[dest].pos.shape[0]

        # Create mask for valid edges (both source and dest indices must be in range)
        valid_src_mask = (edge_idx[0] >= 0) & (edge_idx[0] < num_source_nodes)
        valid_dst_mask = (edge_idx[1] >= 0) & (edge_idx[1] < num_dest_nodes)
        valid_mask = valid_src_mask & valid_dst_mask

        # If no valid edges remain, return empty edge_attr
        if not valid_mask.any():
            empty_attr = torch.empty((0, self.edge_dim), device=self.device)
            self.graph[(source, edge, dest)].edge_attr = empty_attr
            # Also update edge_index to be empty
            self.graph[(source, edge, dest)].edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
            return

        # Filter edge indices to only valid ones
        edge_idx_filtered = edge_idx[:, valid_mask]

        # Update the graph's edge_index with filtered version
        self.graph[(source, edge, dest)].edge_index = edge_idx_filtered

        if all_T_w_e is None:
            # No relative transform – direct edge
            pos_dest = self.graph[dest].pos[edge_idx_filtered[1]]
            pos_source = self.graph[source].pos[edge_idx_filtered[0]]
            pos_dest_rot = pos_dest
        else:
            # Edge defined via relative SE(3) transform
            pos_source = self.graph[source].pos[edge_idx_filtered[0]]

            # Also validate transform indices
            num_transforms = all_T_w_e.shape[0]
            valid_transform_src = (edge_idx_filtered[0] >= 0) & (edge_idx_filtered[0] < num_transforms)
            valid_transform_dst = (edge_idx_filtered[1] >= 0) & (edge_idx_filtered[1] < num_transforms)
            valid_transform_mask = valid_transform_src & valid_transform_dst

            if not valid_transform_mask.all():
                # Further filter if some transform indices are invalid
                edge_idx_filtered = edge_idx_filtered[:, valid_transform_mask]
                self.graph[(source, edge, dest)].edge_index = edge_idx_filtered

                if edge_idx_filtered.shape[1] == 0:
                    empty_attr = torch.empty((0, self.edge_dim), device=self.device)
                    self.graph[(source, edge, dest)].edge_attr = empty_attr
                    return

                pos_source = self.graph[source].pos[edge_idx_filtered[0]]

            T_i_j = torch.bmm(
                all_T_e_w[edge_idx_filtered[0]],
                all_T_w_e[edge_idx_filtered[1]],
            )
            pos_dest_rot = torch.bmm(T_i_j[..., :3, :3], pos_source[..., None]).squeeze(-1)
            pos_dest = pos_source + T_i_j[..., :3, 3]

        # -------------------------------------------------
        # Numerical stability (both branches)
        # -------------------------------------------------
        # Replace NaN/Inf with zeros
        pos_source = torch.where(torch.isfinite(pos_source), pos_source,
                                 torch.zeros_like(pos_source))
        pos_dest = torch.where(torch.isfinite(pos_dest), pos_dest,
                               torch.zeros_like(pos_dest))
        pos_dest_rot = torch.where(torch.isfinite(pos_dest_rot), pos_dest_rot,
                                   torch.zeros_like(pos_dest_rot))

        # Clamp to a safe range (values >10 cause sin/cos overflow)
        pos_source = torch.clamp(pos_source, -10.0, 10.0)
        pos_dest = torch.clamp(pos_dest, -10.0, 10.0)
        pos_dest_rot = torch.clamp(pos_dest_rot, -10.0, 10.0)

        pos_diff = pos_dest - pos_source
        pos_rot_diff = pos_dest_rot - pos_source

        # Edge attribute concatenates positional encodings for both diff and rot diff
        self.graph[(source, edge, dest)].edge_attr = torch.cat(
            [self.pos_embd(pos_diff), self.pos_embd(pos_rot_diff)], dim=-1
        )
