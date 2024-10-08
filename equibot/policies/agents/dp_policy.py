import copy
import hydra
import torch
from torch import nn
import open3d as o3d
import pytorch3d.ops as torch3d_ops
import numpy as np

from equibot.policies.utils.diffusion.ema_model import EMAModel
from equibot.policies.utils.diffusion.simple_conditional_unet1d import ConditionalUnet1D
from equibot.policies.utils.diffusion.resnet_with_gn import get_resnet, replace_bn_with_gn


class DPPolicy(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = hidden_dim = cfg.model.hidden_dim
        self.obs_mode = cfg.model.obs_mode
        self.device = device
        if cfg.model.use_obj_pc_encoder:
            from equibot.policies.vision.pointnet_encoder_new import PointNetEncoder
            self.obj_pc = np.asarray(o3d.io.read_point_cloud(cfg.model.obj_path).points)
            self.obj_pc = torch.from_numpy(self.downsample_with_fps(self.obj_pc, 256)).to(cfg.device, dtype=torch.float32)
        elif cfg.model.use_obj_pc_condition:
            from equibot.policies.vision.pointnet_encoder import PointNetEncoder
            self.obj_pc = np.asarray(o3d.io.read_point_cloud(cfg.model.obj_path).points)
            self.obj_pc = torch.from_numpy(self.downsample_with_fps(self.obj_pc, 256)).to(cfg.device, dtype=torch.float32)
        else:
            from equibot.policies.vision.pointnet_encoder import PointNetEncoder

        # |o|o|                             observations: 2
        # | |a|a|a|a|a|a|a|a|               actions executed: 8
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
        self.pred_horizon = cfg.model.pred_horizon
        self.obs_horizon = cfg.model.obs_horizon
        self.action_horizon = cfg.model.ac_horizon

        if hasattr(cfg.model, "num_diffusion_iters"):
            self.num_diffusion_iters = cfg.model.num_diffusion_iters
        else:
            self.num_diffusion_iters = cfg.model.noise_scheduler.num_train_timesteps

        self.num_eef = cfg.env.num_eef
        self.eef_dim = cfg.env.eef_dim
        self.dof = cfg.env.dof
        if cfg.model.obs_mode == "state":
            self.obs_dim = self.num_eef * self.eef_dim
        elif cfg.model.obs_mode == "rgb":
            self.obs_dim = 512 + self.num_eef * self.eef_dim
        else:
            self.obs_dim = hidden_dim + self.num_eef * self.eef_dim
        # if cfg.model.obs_rgb:
        #     self.obs_dim += 512
        self.action_dim = self.dof * cfg.env.num_eef

        if self.obs_mode.startswith("pc"):
            self.encoder = PointNetEncoder(
                h_dim=hidden_dim,
                c_dim=hidden_dim,
                num_layers=cfg.model.encoder.backbone_args.num_layers,
            )
        elif self.obs_mode == "rgb":
            self.encoder = replace_bn_with_gn(get_resnet("resnet18"))
        else:
            self.encoder = nn.Identity()
        # if cfg.model.obs_rgb:
        #     self.rgb_encoder = replace_bn_with_gn(get_resnet("resnet18"))
        global_cond_dim = self.obs_dim*self.obs_horizon
        if cfg.model.use_lstm:
            global_cond_dim += cfg.model.lstm_dim
        if cfg.model.use_obj_pc_condition:
            global_cond_dim += hidden_dim
            
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            diffusion_step_embed_dim=self.obs_dim * self.obs_horizon,
            global_cond_dim=global_cond_dim,
        )

        self.nets = nn.ModuleDict(
            {"encoder": self.encoder, "noise_pred_net": self.noise_pred_net}
        )

        # if cfg.model.obs_rgb:
        #     self.nets.update({"rgb_encoder": self.rgb_encoder})

        if self.cfg.model.use_lstm:
            self.lstm = nn.LSTM(self.obs_dim, cfg.model.lstm_dim) #Theoretically 1 should be enough, but just in case...
            self.nets.update({"lstm": self.lstm})

        self.ema = EMAModel(model=copy.deepcopy(self.nets), power=0.75)

        self._init_torch_compile()

        self.noise_scheduler = hydra.utils.instantiate(cfg.model.noise_scheduler)

        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized DP Policy with {num_parameters} parameters")

    def _init_torch_compile(self):
        if self.cfg.model.use_torch_compile:
            self.encoder_handle = torch.compile(self.encoder)
            self.noise_pred_net_handle = torch.compile(self.noise_pred_net)
            # if self.cfg.model.obs_rgb:
            #     self.rgb_encoder_handle = torch.compile(self.rgb_encoder)
            if self.cfg.model.use_lstm:
                self.lstm_handle = torch.compile(self.lstm)
    
    def downsample_with_fps(self, points: np.ndarray, n_points):
        # fast point cloud sampling using torch3d
        points = torch.from_numpy(points).unsqueeze(0).cuda()
        # remember to only use coord to sample
        _, sampled_indices = torch3d_ops.sample_farthest_points(points=points[...,:3], K=n_points)
        points = points.squeeze(0).cpu().numpy()
        points = points[sampled_indices.squeeze(0).cpu().numpy()]
        return points

    def forward(self, obs, hidden=None, predict_action=True, debug=False):
        # assumes that observation has format:
        # - pc: [BS, obs_horizon, num_pts, 3]
        # - state: [BS, obs_horizon, obs_dim]
        # returns:
        # - action: [BS, pred_horizon, ac_dim]
        pc = obs["pc"]
        state = obs["state"]
        ret = dict()

        if self.obs_mode.startswith("pc"):
            pc = self.pc_normalizer.normalize(pc)
        state = self.state_normalizer.normalize(state)

        pc_shape = pc.shape
        batch_size = pc.shape[0]

        ema_nets = self.ema.averaged_model

        if self.obs_mode == "state":
            z = state
        else:
            if self.obs_mode.startswith("pc"):
                flattened_pc = pc.reshape(batch_size * self.obs_horizon, *pc_shape[-2:])
                if self.cfg.model.use_obj_pc_encoder:
                    z = ema_nets["encoder"](flattened_pc.permute(0, 2, 1), self.obj_pc.permute(1, 2))["global"]
                else:
                    z = ema_nets["encoder"](flattened_pc.permute(0, 2, 1))["global"]
                z = z.reshape(batch_size, self.obs_horizon, -1)
            # if self.cfg.model.obs_rgb :
            #     rgb = obs["rgb"]
            #     rgb_shape = rgb.shape
            #     flattened_rgb = rgb.reshape(
            #         batch_size * self.obs_horizon, *rgb_shape[-3:]
            #     )
            #     z_rgb = ema_nets["rgb_encoder"](flattened_rgb.permute(0, 3, 1, 2))
            #     z_rgb = z_rgb.reshape(batch_size, self.obs_horizon, -1)
            #     z = torch.cat([z, z_rgb], dim=-1)

            z = torch.cat([z, state], dim=-1)
            if self.cfg.model.use_obj_pc_condition:
                z = torch.cat([z, ema_nets["encoder"](self.obj_pc.permute(1, 2))["global"]], dim=-1)
            obs_cond = z.reshape(batch_size, -1)  # (BS, obs_horizon * obs_dim)
            if self.cfg.model.use_lstm:
                if hidden == None:
                    h = torch.zeros(1, batch_size, self.cfg.model.lstm_dim).to(self.cfg.device)
                    c = torch.zeros(1, batch_size, self.cfg.model.lstm_dim).to(self.cfg.device)
                else:
                    h, c = hidden

                _, (h, c) = ema_nets["lstm"](z.permute(1, 0, 2), (h, c))
                ret.update({"h": h, "c": c})
                obs_cond = torch.cat([obs_cond, h[0]], dim=1)
        

        initial_noise_scale = 0.0 if debug else 1.0
        noisy_action = (
            torch.randn((batch_size, self.pred_horizon, self.action_dim)).to(
                self.device
            )
            * initial_noise_scale
        )
        curr_action = noisy_action
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

        for k in self.noise_scheduler.timesteps:
            # predict noise
            noise_pred = ema_nets["noise_pred_net"](
                sample=curr_action, timestep=k, global_cond=obs_cond,
            )

            # inverse diffusion step
            curr_action = self.noise_scheduler.step(
                model_output=noise_pred, timestep=k, sample=curr_action
            ).prev_sample

        ret.update(dict(ac=curr_action))
        return ret

    def step_ema(self):
        self.ema.step(self.nets)
