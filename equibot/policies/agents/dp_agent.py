import numpy as np
import torch
from torch import nn
import wandb

from tqdm import tqdm

from equibot.policies.utils.norm import Normalizer
from equibot.policies.utils.misc import to_torch
from equibot.policies.agents.dp_policy import DPPolicy
from equibot.policies.utils.diffusion.lr_scheduler import get_scheduler


class DPAgent(object):
    def __init__(self, cfg):
        print(f"Initializing DP agent.")
        self.cfg = cfg
        self._init_actor()
        if cfg.mode == "train":
            self.optimizer = torch.optim.AdamW(
                self.actor.nets.parameters(),
                lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay,
            )
            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.optimizer,
                num_warmup_steps=500,
                num_training_steps=cfg.data.dataset.num_training_steps,
            )
        self.device = cfg.device
        self.num_eef = cfg.env.num_eef
        self.dof = cfg.env.dof
        self.num_points = cfg.data.dataset.num_points
        self.obs_mode = cfg.model.obs_mode
        self.ac_mode = cfg.model.ac_mode
        self.obs_horizon = cfg.model.obs_horizon
        self.pred_horizon = cfg.model.pred_horizon
        self.ac_horizon = cfg.model.ac_horizon
        self.shuffle_pc = cfg.data.dataset.shuffle_pc

        self.pc_normalizer = None
        self.state_normalizer = None
        self.ac_normalizer = None

    def _init_actor(self):
        self.actor = DPPolicy(self.cfg, device=self.cfg.device).to(self.cfg.device)
        self.actor.ema.averaged_model.to(self.cfg.device)

    def _init_normalizers(self, batch):
        if self.obs_mode.startswith("pc") and self.pc_normalizer is None:
            flattened_pc = batch["pc"].view(-1, 3)
            self.pc_normalizer = Normalizer(flattened_pc)
            self.actor.pc_normalizer = self.pc_normalizer
            print(f"PC normalization stats: {self.pc_normalizer.stats}")
        if self.state_normalizer is None:
            state = batch["eef_pos"]
            flattened_state = state.view(-1, state.shape[-1])
            self.state_normalizer = Normalizer(flattened_state)
            self.actor.state_normalizer = self.state_normalizer
            print(f"State normalization stats: {self.state_normalizer.stats}")
        if self.ac_normalizer is None:
            gt_action = batch["action"]
            flattened_gt_action = gt_action.view(-1, gt_action.shape[-1])
            self.ac_normalizer = Normalizer(flattened_gt_action)
            print(f"Action normalization stats: {self.ac_normalizer.stats}")

    def train(self, training=True):
        self.actor.nets.train(training)

    def act(self, obs, hidden=None, return_dict=False, debug=False):
        self.train(False)
        for key in obs.keys():
            obs[key] = torch.tensor(obs[key])
        # if not isinstance(obs["pc"], np.ndarray):
        #     obs["pc"] = obs["pc"].cpu().numpy()
        if "state" in obs:
            obs["eef_pos"] = obs["state"]
        if len(obs["eef_pos"].shape) == 3:
            assert len(obs["pc"][0].shape) == 3  # (obs_horizon, N, 3)
            # obs["pc"] = [[x] for x in obs["pc"]]
            for k in obs:
                if k != "pc" and isinstance(obs[k], np.ndarray):
                    obs[k] = obs[k][:, None]
            has_batch_dim = True
        elif len(obs["eef_pos"].shape) == 4:
            assert len(obs["pc"][0][0].shape) == 2  # (obs_horizon, B, N, 3)
            for key in obs.keys():
                if key != "rgb":
                    obs[key] = obs[key].permute(1, 0, 2, 3)
            has_batch_dim = True
        else:
            raise ValueError("Input format not recognized.")

        ac_dim = self.num_eef * self.dof
        batch_size = len(obs["pc"])

        state = obs["eef_pos"].reshape(tuple(obs["eef_pos"].shape[:2]) + (-1,))

        # process the point clouds
        # some point clouds might be invalid
        # if this occurs, exclude these batch items
        xyzs = []
        ac = np.zeros([batch_size, self.pred_horizon, ac_dim])
        if return_dict:
            ac_dict = []
            for i in range(batch_size):
                ac_dict.append(None)
        forward_idxs = list(np.arange(batch_size))
        # for pcs in obs["pc"]:
        #     xyzs.append([])
        #     for batch_idx, xyz in enumerate(pcs):
        #         if not batch_idx in forward_idxs:
        #             xyzs[-1].append(np.zeros((self.num_points, 3)))
        #         elif xyz.shape[0] == 0:
        #             # no points in point cloud, return no-op action
        #             forward_idxs.remove(batch_idx)
        #             xyzs[-1].append(np.zeros((self.num_points, 3)))
        #         elif self.shuffle_pc:
        #             choice = np.random.choice(
        #                 xyz.shape[0], self.num_points, replace=True
        #             )
        #             xyz = xyz[choice, :]
        #             xyzs[-1].append(xyz)
        #         else:
        #             step = xyz.shape[0] // self.num_points
        #             xyz = xyz[::step, :][: self.num_points]
        #             xyzs[-1].append(xyz)

        if len(forward_idxs) > 0:
            # torch_obs = dict(
            #     pc=torch.tensor(np.array(xyzs).swapaxes(0, 1)[forward_idxs])
            #     .to(self.device)
            #     .float(),
            #     state=torch.tensor(state.swapaxes(0, 1)[forward_idxs])
            #     .to(self.device)
            #     .float(),
            # )
            torch_obs = dict(
                pc = obs["pc"].to(self.device).float(),
                state = state.to(self.device).float()
            )
            for k in obs:
                if not k in ["pc", "eef_pos"] and isinstance(obs[k], np.ndarray):
                    torch_obs[k] = (
                        torch.tensor(obs[k].swapaxes(0, 1)[forward_idxs])
                        .to(self.device)
                        .float()
                    )
            raw_ac_dict = self.actor(torch_obs, hidden, debug=debug)
        else:
            raw_ac_dict = torch.zeros(
                (batch_size, self.actor.pred_horizon, self.actor.action_dim)
            ).to(self.actor.device)
        for i, idx in enumerate(forward_idxs):
            if return_dict:
                ac_dict[idx] = {k: v[i] for k, v in raw_ac_dict.items()}
            unnormed_action = (
                self.ac_normalizer.unnormalize(raw_ac_dict["ac"][i])
                .detach()
                .cpu()
                .numpy()
            )
            ac[idx] = unnormed_action
        
        if not has_batch_dim:
            ac = ac[0]
            if return_dict:
                ac_dict = ac_dict[0]
        if return_dict:
            return ac, ac_dict
        elif hidden is not None:
            return ac, raw_ac_dict["h"], raw_ac_dict["c"]
        else:
            return ac

    def update(self, batch, vis=False):
        self.train()

        batch = to_torch(batch, self.device)
        batch["eef_pos"] = batch["eef_pos"].reshape(
            tuple(batch["eef_pos"].shape[:2]) + (-1,)
        )
        pc = batch["pc"]
        # rgb = batch["rgb"]
        state = batch["eef_pos"]
        gt_action = batch["action"]

        if self.state_normalizer is None or self.ac_normalizer is None:
            self._init_normalizers(batch)
        if self.obs_mode.startswith("pc"):
            pc = self.pc_normalizer.normalize(pc)
        state = self.state_normalizer.normalize(state)
        gt_action = self.ac_normalizer.normalize(gt_action)

        pc_shape = pc.shape
        batch_size = pc.shape[0]
        # rgb_shape = rgb.shape

        if self.obs_mode == "state":
            z = state
        else:
            flattened_pc = pc.reshape(batch_size * self.obs_horizon, *pc_shape[-2:])
            if self.cfg.model.use_torch_compile:
                if self.cfg.model.use_obj_pc_encoder:
                    z = self.actor.encoder_handle(flattened_pc.permute(0, 2, 1), self.actor.obj_pc.permute(1,0))["global"]
                else:
                    z = self.actor.encoder_handle(flattened_pc.permute(0, 2, 1))["global"]
            else:
                if self.cfg.model.use_obj_pc_encoder:
                    z = self.actor.encoder(flattened_pc.permute(0, 2, 1), self.actor.obj_pc.permute(1,0))["global"]
                else:
                    z = self.actor.encoder(flattened_pc.permute(0, 2, 1))["global"]

            z = z.reshape(batch_size, self.obs_horizon, -1)
            # if self.cfg.model.obs_rgb:
            #     flattened_rgb = rgb.reshape(
            #         batch_size * self.obs_horizon, *rgb_shape[-3:]
            #     )
            #     z_rgb = self.actor.rgb_encoder(flattened_rgb.permute(0, 3, 1, 2))
            #     z_rgb = z_rgb.reshape(batch_size, self.obs_horizon, -1)
            #     z = torch.cat([z, z_rgb], dim=-1)
            z = torch.cat([z, state], dim=-1).reshape(batch_size, -1)
            if self.cfg.model.use_obj_pc_condition:
                z_obj = self.actor.encoder(self.actor.obj_pc.permute(1, 0))["global"]
                z_obj = z_obj.repeat(batch_size, 1)
                z = torch.cat([z, z_obj], dim=-1)

        obs_cond = z.reshape(batch_size, -1)  # (BS, obs_horizon * obs_dim)

        noise = torch.randn(gt_action.shape, device=self.device)

        timesteps = torch.randint(
            0,
            self.actor.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.device,
        ).long()

        noisy_actions = self.actor.noise_scheduler.add_noise(
            gt_action, noise, timesteps
        )

        if self.cfg.model.use_torch_compile:
            noise_pred = self.actor.noise_pred_net_handle(
                noisy_actions, timesteps, global_cond=obs_cond
            )
        else:
            noise_pred = self.actor.noise_pred_net(
                noisy_actions, timesteps, global_cond=obs_cond
            )

        loss = nn.functional.mse_loss(noise_pred, noise)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        # make sure gradients are complete
        if self.obs_mode == "state":
            grads = [k.grad for k in self.actor.noise_pred_net.parameters()]
        else:
            grads = self.actor.nets.parameters()
        assert np.array([g is not None for g in grads]).all()

        self.actor.step_ema()

        metrics = {
            "loss": loss.item(),
            "mean_gt_noise_norm": np.linalg.norm(
                noise.reshape(gt_action.shape[0], -1).detach().cpu().numpy(), axis=1
            ).mean(),
            "mean_pred_noise_norm": np.linalg.norm(
                noise_pred.reshape(gt_action.shape[0], -1).detach().cpu().numpy(),
                axis=1,
            ).mean(),
        }
        return metrics
    
    def update_lstm(self, batch, vis=False):
        self.train()

        batch = to_torch(batch, self.device)
        batch["eef_pos"] = batch["eef_pos"].reshape(
            tuple(batch["eef_pos"].shape[:2]) + (-1,)
        )
        pc_all = batch["pc"]
        # rgb_all = batch["rgb"]
        state_all = batch["eef_pos"]
        gt_action_all = batch["action"]
        length = batch["length"]

        if self.state_normalizer is None or self.ac_normalizer is None:
            self._init_normalizers(batch)
        if self.obs_mode.startswith("pc"):
            pc_all = self.pc_normalizer.normalize(pc_all)
        state_all = self.state_normalizer.normalize(state_all)
        gt_action_all = self.ac_normalizer.normalize(gt_action_all)

        batch_size = pc_all.shape[0]

        h = torch.zeros(1, batch_size, self.cfg.model.lstm_dim, device=self.cfg.device)
        c = torch.zeros(1, batch_size, self.cfg.model.lstm_dim, device=self.cfg.device)

        losses = []
        losses_update = []
        gt_noise_norms = []
        pred_noise_norms = []

        for ep_t in tqdm(range(self.cfg.model.obs_horizon-1, pc_all.shape[1]-self.cfg.model.pred_horizon),
            leave=False):
            start_t = ep_t - (self.cfg.model.obs_horizon - 1)
            end_t = ep_t + self.cfg.model.pred_horizon

            indices = torch.where(length > end_t)[0]
            pc = pc_all[indices, start_t:ep_t+1]
            # rgb = rgb_all[indices, start_t:ep_t+1]
            state = state_all[indices, start_t:ep_t+1]
            action = gt_action_all[indices, ep_t+1:end_t+1]

            pc_shape = pc.shape
            # rgb_shape = rgb.shape

            if self.obs_mode == "state":
                z = state
            else:
                flattened_pc = pc.reshape(pc_shape[0] * pc_shape[1], *pc_shape[-2:])
                if self.cfg.model.use_torch_compile:
                    if self.cfg.model.use_obj_pc_encoder:
                        z = self.actor.encoder_handle(flattened_pc.permute(0, 2, 1), self.actor.obj_pc)["global"]
                    else:
                        z = self.actor.encoder_handle(flattened_pc.permute(0, 2, 1))["global"]
                else:
                    if self.cfg.model.use_obj_pc_encoder:
                        z = self.actor.encoder(flattened_pc.permute(0, 2, 1), self.actor.obj_pc)["global"]
                    else:
                        z = self.actor.encoder(flattened_pc.permute(0, 2, 1))["global"]

                z = z.reshape(pc_shape[0], pc_shape[1], -1)
                # if self.cfg.model.obs_rgb:
                #     flattened_rgb = rgb.reshape(
                #         batch_size * self.obs_horizon, *rgb_shape[-3:]
                #     )
                #     z_rgb = self.actor.rgb_encoder(flattened_rgb.permute(0, 3, 1, 2))
                #     z_rgb = z_rgb.reshape(batch_size, self.obs_horizon, -1)
                #     z = torch.cat([z, z_rgb], dim=-1)

                z = torch.cat([z, state], dim=-1).reshape(pc_shape[0], -1)
                if self.cfg.model.use_obj_pc_condition:
                    z_obj = self.actor.encoder(self.actor.obj_pc.permute(1, 0))["global"]
                    z_obj = z_obj.repeat(batch_size, 1)
                    z = torch.cat([z, z_obj], dim=-1)
                
                pc_cur = pc[:, -1]
                if self.cfg.model.use_torch_compile:
                    if self.cfg.model.use_obj_pc_encoder:
                        z_cur = self.actor.encoder_handle(pc_cur.permute(0, 2, 1), self.actor.obj_pc)["global"]
                    else:
                        z_cur = self.actor.encoder_handle(pc_cur.permute(0, 2, 1))["global"]
                else:
                    if self.cfg.model.use_obj_pc_encoder:
                        z_cur = self.actor.encoder(pc_cur.permute(0, 2, 1), self.actor.obj_pc)["global"]
                    else:
                        z_cur = self.actor.encoder(pc_cur.permute(0, 2, 1))["global"]

                z_cur = torch.cat([z_cur, state[:, -1]], dim=-1).reshape(1, pc_shape[0], -1)
                
                if self.cfg.model.use_torch_compile:
                    _, (h[:, indices], c[:, indices]) = self.actor.lstm_handle(z_cur, (h[:, indices], c[:, indices]))
                else:
                    _, (h[:, indices], c[:, indices]) = self.actor.lstm(z_cur, (h[:, indices], c[:, indices]))

            obs_cond = z.reshape(pc_shape[0], -1)  # (BS, obs_dim * obs_horizon)
            obs_cond = torch.cat([obs_cond, h[0, indices]], dim=1)

            noise = torch.randn(action.shape, device=self.device)

            timesteps = torch.randint(
                0,
                self.actor.noise_scheduler.config.num_train_timesteps,
                (pc_shape[0],),
                device=self.device,
            ).long()

            noisy_actions = self.actor.noise_scheduler.add_noise(
                action, noise, timesteps
            )

            if self.cfg.model.use_torch_compile:
                noise_pred = self.actor.noise_pred_net_handle(
                    noisy_actions, timesteps, global_cond=obs_cond
                )
            else:
                noise_pred = self.actor.noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond
                )

            loss = nn.functional.mse_loss(noise_pred, noise)
            losses.append(loss.item())
            losses_update.append(loss)

            if len(losses_update) == self.cfg.training.loss_every:
                self.optimizer.zero_grad()
                torch.sum(torch.stack(losses_update)).backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                losses_update = []

            # make sure gradients are complete
            if self.obs_mode == "state":
                grads = [k.grad for k in self.actor.noise_pred_net.parameters()]
            else:
                grads = self.actor.nets.parameters()
            assert np.array([g is not None for g in grads]).all()

            self.actor.step_ema()

            h = h.detach()
            c = c.detach()

            gt_noise_norms.append(np.linalg.norm(
                noise.reshape(action.shape[0], -1).detach().cpu().numpy(), axis=1
                ).mean())
            pred_noise_norms.append(np.linalg.norm(
                noise_pred.reshape(action.shape[0], -1).detach().cpu().numpy(), axis=1,
                ).mean())

        metrics = {
            "loss": np.mean(losses),
            "mean_gt_noise_norm": np.mean(gt_noise_norms),
            "mean_pred_noise_norm": np.mean(pred_noise_norms),
        }
        return metrics


    def visualize_sample(
        self,
        rgb,
        ac_dict,
        eef_pos,
        gt_offset=None,
        offset_weights=None,
        global_features=None,
        return_wandb_image=True,
    ):
        vis_rgb = rgb.copy()
        if return_wandb_image:
            vis_rgb = wandb.Image(vis_rgb)
        return vis_rgb

    def save_snapshot(self, save_path):
        state_dict = dict(
            actor=self.actor.state_dict(),
            ema_model=self.actor.ema.averaged_model.state_dict(),
            pc_normalizer=self.pc_normalizer.state_dict(),
            state_normalizer=self.state_normalizer.state_dict(),
            ac_normalizer=self.ac_normalizer.state_dict(),
        )
        torch.save(state_dict, save_path)

    def _fix_state_dict_keys(self, state_dict):
        return {k: v for k, v in state_dict.items() if not "handle" in k}

    def load_snapshot(self, save_path):
        state_dict = torch.load(save_path)
        self.state_normalizer = Normalizer(state_dict["state_normalizer"])
        self.actor.state_normalizer = self.state_normalizer
        self.ac_normalizer = Normalizer(state_dict["ac_normalizer"])
        if self.obs_mode.startswith("pc"):
            self.pc_normalizer = Normalizer(state_dict["pc_normalizer"])
            self.actor.pc_normalizer = self.pc_normalizer
        if hasattr(self, "encoder_handle"):
            del self.encoder_handle
            del self.noise_pred_net_handle
        self.actor.load_state_dict(self._fix_state_dict_keys(state_dict["actor"]))
        self.actor._init_torch_compile()
        self.actor.ema.averaged_model.load_state_dict(state_dict["ema_model"])
