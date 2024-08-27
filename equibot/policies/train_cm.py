if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import itertools
import os

import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time

import threading
import torch.nn.functional as F
from typing import Generator
from hydra.core.hydra_config import HydraConfig
from equibot.policies.utils.misc import get_dataset, get_agent
#from diffusion_policy_3d.env_runner.base_runner import BaseRunner
#from equibot.policies.utils.checkpoint_util import TopKCheckpointManager
from equibot.policies.utils.pytorch_util import dict_apply, optimizer_to
from equibot.policies.utils.diffusion.ema_model import EMAModel
from equibot.policies.utils.diffusion.lr_scheduler import get_scheduler
#from equibot.policies.utils.equivariant_diffusion.simple_conditional_unet1d import ConditionalUnet1D
from equibot.policies.utils.diffusion.simple_conditional_unet1d import ConditionalUnet1D

OmegaConf.register_new_resolver("eval", eval, replace=True)


@torch.no_grad()
def update_ema(target_params: Generator, source_params: Generator, rate: float = 0.99) -> None:
    for tgt, src in zip(target_params, source_params):
        tgt.detach().mul_(rate).add_(src, alpha=1 - rate)

def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def predicted_origin(
        model_output: torch.Tensor,
        timesteps: torch.Tensor,
        sample: torch.Tensor,
        prediction_type: str,
        alphas: torch.Tensor,
        sigmas: torch.Tensor
) -> torch.Tensor:
    if prediction_type == "epsilon":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
        alphas = extract_into_tensor(alphas, timesteps, sample.shape)
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(f"Prediction type {prediction_type} currently not supported.")

    return pred_x_0

class DDIMSolver:
    def __init__(self, alpha_cumprods: np.ndarray, timesteps: int = 1000, ddim_timesteps: int = 50) -> None:
        # DDIM sampling parameters
        step_ratio = timesteps // ddim_timesteps
        self.ddim_timesteps = (np.arange(1, ddim_timesteps + 1) * step_ratio).round().astype(np.int64) - 1
        self.ddim_alpha_cumprods = alpha_cumprods[self.ddim_timesteps]
        self.ddim_alpha_cumprods_prev = np.asarray(
            [alpha_cumprods[0]] + alpha_cumprods[self.ddim_timesteps[:-1]].tolist()
        )
        # convert to torch tensors
        self.ddim_timesteps = torch.from_numpy(self.ddim_timesteps).long()
        self.ddim_alpha_cumprods = torch.from_numpy(self.ddim_alpha_cumprods)
        self.ddim_alpha_cumprods_prev = torch.from_numpy(self.ddim_alpha_cumprods_prev)

    def to(self, device: torch.device) -> "DDIMSolver":
        self.ddim_timesteps = self.ddim_timesteps.to(device)
        self.ddim_alpha_cumprods = self.ddim_alpha_cumprods.to(device)
        self.ddim_alpha_cumprods_prev = self.ddim_alpha_cumprods_prev.to(device)
        return self

    def ddim_step(self, pred_x0: torch.Tensor, pred_noise: torch.Tensor,
                  timestep_index: torch.Tensor) -> torch.Tensor:
        alpha_cumprod_prev = extract_into_tensor(self.ddim_alpha_cumprods_prev, timestep_index, pred_x0.shape)
        dir_xt = (1.0 - alpha_cumprod_prev).sqrt() * pred_noise
        x_prev = alpha_cumprod_prev.sqrt() * pred_x0 + dir_xt
        return x_prev

class TrainWorkspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        
        # set seed
        seed = cfg.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.dataset = get_dataset(cfg, "train")

        batch_size = cfg.training.batch_size
        cfg.data.dataset.num_training_steps = (
            cfg.training.num_epochs * len(self.dataset) // batch_size
        )

        # configure model
        self.model = get_agent(cfg.agent.agent_name)(cfg)

        self.ema_model = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except: # minkowski engine could not be copied. recreate it
                self.ema_model = get_agent(cfg.agent.agent_name)(cfg)

        # configure training state
        self.global_step = 0
        self.epoch = 0
    
    def update(self, cfg, batch, unet, teacher_unet, target_unet, hidden=None):
        # normalize input
        pc = batch["pc"]
        state = batch["eef_pos"]
        action = batch["action"]

        self.model.pc_normalizer.to(cfg.device)
        self.model.state_normalizer.to(cfg.device)
        self.model.ac_normalizer.to(cfg.device)

        if self.model.state_normalizer is None or self.model.ac_normalizer is None:
            self.model._init_normalizers(batch)
        pc = self.model.pc_normalizer.normalize(pc)
        state = self.model.state_normalizer.normalize(state)
        action = self.model.ac_normalizer.normalize(action)
        
        batch_size = pc.shape[0]
        pc_shape = pc.shape
        horizon = self.cfg.model.obs_horizon

        # handle different ways of passing observation
        trajectory = action

        if self.model.obs_mode == "state":
            z = state
        else:
            assert self.model.obs_mode != "rgb"
            flattened_pc = pc.reshape(batch_size * cfg.model.obs_horizon, *pc_shape[-2:])
            if self.cfg.model.use_torch_compile:
                z = self.model.actor.encoder_handle(flattened_pc.permute(0, 2, 1))["global"]
            else:
                z = self.model.actor.encoder(flattened_pc.permute(0, 2, 1))["global"]

            z = z.reshape(batch_size, horizon, -1)
            z = torch.cat([z, state], dim=-1)

            if cfg.model.use_lstm:
                pc_cur = pc[:, -1]
                with torch.no_grad():
                    if self.cfg.model.use_torch_compile:
                        z_cur = self.model.actor.encoder_handle(pc_cur.permute(0, 2, 1))["global"]
                    else:
                        z_cur = self.model.actor.encoder(pc_cur.permute(0, 2, 1))["global"]

                    z_cur = torch.cat([z_cur, state[:, -1]], dim=-1).reshape(1, batch_size, -1)
                    
                if self.cfg.model.use_torch_compile:
                    _, hidden = self.model.actor.lstm_handle(z_cur, hidden)
                else:
                    _, hidden = self.model.actor.lstm(z_cur, hidden)
        
        obs_cond = z.reshape(batch_size, -1)  # (BS, obs_horizion * obs_dim)
        if cfg.model.use_lstm:
            obs_cond = torch.cat([obs_cond, hidden[0][0]], dim=1)

        noise = torch.randn(trajectory.shape, device=trajectory.device)

        latents = trajectory

        # Sample a random timestep for each image t_n ~ U[0, N - k - 1] without bias.
        topk = self.noise_scheduler.config.num_train_timesteps // self.model.actor.num_diffusion_iters
        index = torch.randint(0, self.model.actor.num_diffusion_iters, (batch_size,), device=cfg.device).long()
        start_timesteps = self.solver.ddim_timesteps[index]
        timesteps = start_timesteps - topk
        timesteps = torch.where(timesteps < 0, torch.zeros_like(timesteps), timesteps)

        # 20.4.4. Get boundary scalings for start_timesteps and (end) timesteps.
        c_skip_start, c_out_start = scalings_for_boundary_conditions(start_timesteps)
        c_skip_start, c_out_start = [append_dims(x, latents.ndim) for x in [c_skip_start, c_out_start]]
        c_skip, c_out = scalings_for_boundary_conditions(timesteps)
        c_skip, c_out = [append_dims(x, latents.ndim) for x in [c_skip, c_out]]

        noisy_model_input = self.noise_scheduler.add_noise(latents, noise, start_timesteps)
        noise_pred = unet(
            sample=noisy_model_input, 
            timestep=start_timesteps,
            global_cond=obs_cond)
        pred_x_0 = predicted_origin(
            noise_pred,
            start_timesteps,
            noisy_model_input,
            self.noise_scheduler.config.prediction_type,
            self.alpha_schedule,
            self.sigma_schedule)
        model_pred = c_skip_start * noisy_model_input + c_out_start * pred_x_0

        with torch.no_grad():
            cond_teacher_output = teacher_unet(
                sample=noisy_model_input, 
                timestep=start_timesteps,
                global_cond=obs_cond)
            
            cond_pred_x0 = predicted_origin(
                cond_teacher_output,
                start_timesteps,
                noisy_model_input,
                self.noise_scheduler.config.prediction_type,
                self.alpha_schedule,
                self.sigma_schedule)
            
            x_prev = self.solver.ddim_step(cond_pred_x0, cond_teacher_output, index)

        with torch.no_grad():
            target_noise_pred = target_unet(
                x_prev.float(),
                timesteps,
                global_cond=obs_cond)
            pred_x_0 = predicted_origin(
                target_noise_pred,
                timesteps,
                x_prev,
                self.noise_scheduler.config.prediction_type,
                self.alpha_schedule,
                self.sigma_schedule)
            target = c_skip * x_prev + c_out * pred_x_0

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(), cfg.training.max_grad_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        update_ema(target_unet.parameters(), unet.parameters(), cfg.training.ema_decay)
        
        if cfg.model.use_lstm:
            return loss, hidden
        return loss

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        
        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        RUN_VALIDATION = False # reduce time cost
        
        # resume training
        if cfg.checkpoint.path is not None:
            print(f"Resuming from checkpoint {cfg.checkpoint.path}")
            self.load_checkpoint()
            start_epoch_ix = int(cfg.checkpoint.path.split("/")[-1].split(".")[0][4:])
        else:
            start_epoch_ix = 0

        # configure dataloader
        batch_size = cfg.training.batch_size
        num_workers = cfg.data.dataset.num_workers
        if cfg.model.use_lstm:
            def custom_collate(data):
                pc = [torch.tensor(d['pc']) for d in data]
                eef_pos = [torch.tensor(d['eef_pos']) for d in data]
                action = [torch.tensor(d['action']) for d in data]
                length = torch.tensor([len(pc_item) for pc_item in pc])

                pc = torch.nn.utils.rnn.pad_sequence(pc, batch_first=True)
                eef_pos = torch.nn.utils.rnn.pad_sequence(eef_pos, batch_first=True)
                action = torch.nn.utils.rnn.pad_sequence(action, batch_first=True)
                
                return dict(pc=pc, rgb=rgb, eef_pos=eef_pos, action=action, length=length)
                            
            train_dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
                collate_fn=custom_collate,
            )
        else:
            train_dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
            )
    
        # device transfer
        device = torch.device(cfg.device)
        self.model.actor.to(device)
        self.model.actor.ema.averaged_model.to(device)

        self.noise_scheduler = self.model.actor.noise_scheduler
        self.alpha_schedule = torch.sqrt(self.noise_scheduler.alphas_cumprod).to(device)
        self.sigma_schedule = torch.sqrt(1 - self.noise_scheduler.alphas_cumprod).to(device)

        self.solver = DDIMSolver(
            self.noise_scheduler.alphas_cumprod.numpy(),
            timesteps=self.noise_scheduler.config.num_train_timesteps,
            ddim_timesteps=self.model.actor.num_diffusion_iters,
        ).to(device)

        encoder = self.model.actor.encoder
        unet = self.model.actor.noise_pred_net

        encoder.requires_grad_(False)

        teacher_unet = copy.deepcopy(unet)
        target_unet = copy.deepcopy(unet)

        teacher_unet = teacher_unet.to(device)
        target_unet = target_unet.to(device)
        teacher_unet.requires_grad_(False)
        target_unet.requires_grad_(False)

        # Non-learning normalizers
        # self.model.pc_normalizer.requires_grad_(False)
        # self.model.state_normalizer.requires_grad_(False)
        # self.model.ac_normalizer.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(
            # itertools.chain(unet.parameters(), self.model.condition_attention.parameters()),
            unet.parameters(),
            lr=cfg.training.lr,
            betas=(cfg.training.betas[0], cfg.training.betas[1]),
            weight_decay=cfg.training.weight_decay,
            eps=cfg.training.eps)
        
        self.lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every
        )

        # save batch for sampling
        train_sampling_batch = None

        if cfg.use_wandb:
            wandb_config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=False
            )
            wandb_run = wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                tags=["train_cm"],
                name=cfg.prefix,
                settings=wandb.Settings(code_dir="."),
                config=wandb_config,
            )

        self.global_step = 0
        self.epoch = 0
        
        # training loop
        for _ in tqdm.tqdm(range(start_epoch_ix, cfg.training.num_epochs)):
            step_log = dict()
            # ========= train for this epoch ==========
            train_losses = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                    leave=False, mininterval=1.0) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    batch["eef_pos"] = batch["eef_pos"].reshape(
                        tuple(batch["eef_pos"].shape[:2]) + (-1,))
                    if train_sampling_batch is None:
                        train_sampling_batch = batch
                    
                    if cfg.model.use_lstm:
                        length = batch["length"]
                        h = torch.zeros(1, batch_size, self.cfg.model.lstm_dim, device=self.cfg.device)
                        c = torch.zeros(1, batch_size, self.cfg.model.lstm_dim, device=self.cfg.device)

                        for ep_t in tqdm.tqdm(range(cfg.model.obs_horizon-1,
                            batch["pc"].shape[1]-cfg.model.pred_horizon), leave=False):
                            
                            start_t = ep_t - (cfg.model.obs_horizon - 1)
                            end_t = ep_t + cfg.model.pred_horizon

                            indices = torch.where(length > end_t)[0]
                            batch_sliced = dict(pc=batch["pc"][indices, start_t:ep_t+1],
                            eef_pos=batch["eef_pos"][indices, start_t:ep_t+1],
                            action = batch["action"][indices, ep_t+1:end_t+1])

                            loss, (h[:, indices], c[:, indices]) = self.update(cfg,
                                batch_sliced, unet, teacher_unet, target_unet,
                                hidden=(h[:, indices], c[:, indices]))
                            h = h.detach()
                            c = c.detach()

                            # logging
                            raw_loss_cpu = loss.item()
                            tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                            train_losses.append(raw_loss_cpu)
                            step_log = {
                                'train_loss': raw_loss_cpu,
                                'global_step': self.global_step,
                                'epoch': self.epoch,
                                'lr': self.lr_scheduler.get_last_lr()[0]
                            }
                            
                            loss_dict = {'bc_loss': loss.item()}
                            step_log.update(loss_dict)

                    else:
                        loss = self.update(cfg, batch, unet, teacher_unet, target_unet)
                    
                        # logging
                        raw_loss_cpu = loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': self.lr_scheduler.get_last_lr()[0]
                        }
                        loss_dict = {'bc_loss': loss.item()}
                        step_log.update(loss_dict)


            # at the end of each epoch
            # replace train_loss with epoch average
            train_loss = np.mean(train_losses)
            step_log['train_loss'] = train_loss

            
            # ========= eval for this epoch ==========
            policy = self.model
            policy.train(False)

            # run diffusion sampling on a training batch
            if (self.epoch % cfg.training.sample_every) == 0:
                with torch.no_grad():
                    if cfg.model.use_lstm and train_sampling_batch["pc"].shape[1] > cfg.model.obs_horizon:
                        ep_t = cfg.model.obs_horizon-1
                        end_t = ep_t + cfg.model.pred_horizon
                        indices = torch.where(length > end_t)[0]
                        train_sampling_batch = dict(pc=train_sampling_batch["pc"][indices, :ep_t+1],
                            eef_pos=train_sampling_batch["eef_pos"][indices, :ep_t+1],
                            action=train_sampling_batch["action"][indices, ep_t+1:end_t+1])
                    # sample trajectory from training set, and evaluate difference
                    batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                    
                    pred_action = torch.from_numpy(self.model.act(batch)).to(device)
                    mse = torch.nn.functional.mse_loss(pred_action, batch['action'])
                    step_log['train_action_mse_error'] = mse.item()
                    del batch
                    del pred_action
                    del mse
                
            # checkpoint
            if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                # checkpointing
                if cfg.checkpoint.save_last_ckpt:
                    self.save_checkpoint(start_epoch_ix + self.epoch)
                if cfg.checkpoint.save_last_snapshot:
                    self.save_snapshot()

                # sanitize metric names
                metric_dict = dict()
                for key, value in step_log.items():
                    new_key = key.replace('/', '_')
                    metric_dict[new_key] = value
                
                # We can't copy the last checkpoint here
                # since save_checkpoint uses threads.
                # therefore at this point the file might have been empty!
                # topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                # if topk_ckpt_path is not None:
                #     self.save_checkpoint(path=topk_ckpt_path)
            # ========= eval end for this epoch ==========
            policy.train(True)
            

            # end of epoch
            # log of last step is combined with validation and rollout
            if cfg.use_wandb:
                wandb_run.log(step_log, step=self.global_step)
            self.global_step += 1
            self.epoch += 1
            #del step_log
        
        if cfg.checkpoint.save_last_ckpt:
            self.save_checkpoint(start_epoch_ix + self.epoch)
        
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            # output_dir = HydraConfig.get().runtime.output_dir
            output_dir = os.path.join(os.getcwd(), self.cfg.prefix)
            os.makedirs(output_dir, exist_ok=True)
        return output_dir
    

    def save_checkpoint(self, epoch, path=None):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath(f"ckpt{epoch:05d}.pth")
        else:
            path = pathlib.Path(path)
        os.makedirs(path.parent, exist_ok=True)
        self.model.save_snapshot(path)
    
    # def get_checkpoint_path(self, tag='latest'):
    #     if tag=='latest' or tag=='newest':
    #         return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
    #     elif tag=='best': 
    #         # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
    #         # find the best checkpoint
    #         checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
    #         all_checkpoints = os.listdir(checkpoint_dir)
    #         best_ckpt = None
    #         best_score = -1e10
    #         for ckpt in all_checkpoints:
    #             if 'latest' in ckpt:
    #                 continue
    #             score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
    #             if score > best_score:
    #                 best_ckpt = ckpt
    #                 best_score = score
    #         return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
    #     else:
    #         raise NotImplementedError(f"tag {tag} not implemented")
    
    def load_checkpoint(self, **kwargs):
        # if path is None:
        #     path = self.get_checkpoint_path(tag=tag)
        # else:
        #     path = pathlib.Path(path)
        self.model.load_snapshot(os.path.join(os.getcwd(), self.cfg.checkpoint.path))
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)

@hydra.main(
    version_base=None,
    config_path='configs'
)
def main(cfg):
    workspace = TrainWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
