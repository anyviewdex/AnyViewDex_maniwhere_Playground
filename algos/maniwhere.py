# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
from torchvision.models import resnet
import utils
import random
from collections import deque
from utils import random_overlay, random_mask_freq_v2

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

def info_nce_loss(features1, features2, temperature=0.1):
    features1 = torch.nn.functional.normalize(features1, dim=1)
    features2 = torch.nn.functional.normalize(features2, dim=1)
    sim_matrix = torch.matmul(features1, features2.T) / temperature
    labels = torch.arange(features1.size(0), dtype=torch.long, device=features1.device)
    return torch.nn.functional.cross_entropy(sim_matrix, labels)

class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        self.repr_dim = 256
        self.device = "cuda"
        # Calculate the number of stacked frames (e.g., 9 channels / 3 = 3 frames)
        self.num_frames = obs_shape[0] // 3

        # Initialize full resnet
        resnet = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        ).to(torch.bfloat16)
        
        # Strip the avgpool and fc layers to keep the 4D spatial grid natively
        # This extracts all layers up to layer4, returning (B, 512, H_out, W_out)
        self.resnet18 = nn.Sequential(*list(resnet.children())[:-2]).cuda()

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).cuda())
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1).cuda())

        with torch.no_grad():
            x = torch.randn((1, *obs_shape)).to(torch.float32).cuda()
            B, C, H, W = x.shape
            
            # Fold frame stack into batch dimension to match ImageNet 3-channel expectation
            x = x.view(B * self.num_frames, 3, H, W)
            x = x / 255.0
            x = (x - self.mean) / self.std
            x = x.to(torch.bfloat16)
            
            grid = self.resnet18(x)
            out_shape = grid.shape
        
        # The flattened dimension accounts for the spatial grid of ALL frames
        # out_shape is (B*num_frames, 512, H_out, W_out)
        self.flattened_dim = out_shape[1] * out_shape[2] * out_shape[3] * self.num_frames

        self.visual_proj = nn.Sequential(
            nn.Linear(self.flattened_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU()
        ).cuda()

        self.infonce_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128)
        ).cuda()

    def forward(self, obs, return_infonce=False):
        B, C, H, W = obs.shape
        
        # Fold frame stack into batch dimension
        x = obs.view(B * self.num_frames, 3, H, W)

        x = x.to(torch.float32) / 255.0
        x = (x - self.mean) / self.std
        x = x.to(torch.bfloat16)

        # Safely returns 4D tensor (B*num_frames, 512, H_out, W_out)
        grid = self.resnet18(x).to(torch.float32)
        
        # Flatten spatial dimensions: (B*num_frames, 512 * H_out * W_out)
        flat_visual = grid.flatten(start_dim=1)
        # Unfold frame stack: (B, num_frames * 512 * H_out * W_out)
        flat_visual = flat_visual.view(B, -1)
        
        img_features = self.visual_proj(flat_visual)

        if return_infonce:
            embed = self.infonce_proj(grid) # (B*num_frames, 128)
            embed = embed.view(B, -1)       # Unfold frame stack for info_nce loss: (B, num_frames * 128)
            return img_features, embed
            
        return img_features

class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist
    

class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2



class SigmoidLR:
    def __init__(self, optimizer, lr_max=1, lr_min=0, sigmoid_slope=0.015, sigmoid_center=500):
        self.optimizer = optimizer
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.sigmoid_slope = sigmoid_slope
        self.sigmoid_center = sigmoid_center
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)

    def lr_lambda(self, epoch):
        lr = (self.lr_max - self.lr_min) / (1 + np.exp(-self.sigmoid_slope * (epoch - self.sigmoid_center))) + self.lr_min
        return lr

    def step(self):
        self.scheduler.step()

class ManiAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb, use_wandb,
                 temp, aux_coef, aux_l2_coef, aux_tcc_coef, aux_latency):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb or use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.aux_coef = aux_coef
        self.aux_l2_coef = aux_l2_coef
        self.aux_tcc_coef = aux_tcc_coef
        self.aux_latency = aux_latency

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.obj_pos_head = nn.Sequential(
            nn.Linear(self.encoder.repr_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3) 
        ).to(device)
        self.obj_pos_opt = torch.optim.Adam(self.obj_pos_head.parameters(), lr=lr)

        self.auxiliary_temp = temp
        if int(torch.__version__.split('.')[0]) >= 2:
            self.encoder = torch.compile(self.encoder)
            self.actor = torch.compile(self.actor)
            self.critic = torch.compile(self.critic)
            self.critic_target = torch.compile(self.critic_target)

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.encoder_aux_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.obj_pos_head.train(training)

    def act(self, obs, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]
    
    def read_q(self, obs, step):
        stddev = utils.schedule(self.stddev_schedule, step)
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        dist = self.actor(obs, stddev)
        action = dist.mean
        q = self.critic(obs, action)
        
        return q


    def update_critic(self, obs, action, reward, discount, next_obs, step, aug_obs, aug_move_obs):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        aug_Q1, aug_Q2 = self.critic(aug_obs, action)
        aug_loss = F.mse_loss(aug_Q1, target_Q) + F.mse_loss(aug_Q2, target_Q)
        
        
        # critic_loss = 0.5 * (critic_loss + aug_loss) + 0.3 * aug_move_loss
        if step > self.aux_latency:
            aug_move_Q1, aug_move_Q2 = self.critic(aug_move_obs, action)
            aug_move_loss = F.mse_loss(aug_move_Q1, target_Q) + F.mse_loss(aug_move_Q2, target_Q)
            critic_loss = 0.5 * critic_loss + 0.25 * (aug_loss + aug_move_loss)
            # critic_loss = 0.5 * (critic_loss + aug_move_loss)
            # critic_loss = 0.5 * (critic_loss + aug_loss) + 0.3 * aug_move_loss
        else:
            critic_loss = 0.5 * (critic_loss + aug_loss)
        # critic_loss = 0.5 * (critic_loss + aug_move_loss)

        # l2_loss_aug = F.mse_loss(obs, aug_obs) * 0.1

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        # (critic_loss + l2_loss_aug).backward()
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()
        
        if self.use_tb:
            # Access index 0 because resnet18 is a Sequential wrapper
            grad = self.encoder.resnet18[0].weight.grad
            metrics['grad_critic_mean'] = grad.mean().item() if grad is not None else 0
            metrics['grad_critic_max'] = grad.max().item() if grad is not None else 0
            metrics['grad_critic_min'] = grad.min().item() if grad is not None else 0

        return metrics

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update_auxiliary(self, step, fix_obs, move_obs, obj_pos):
        metrics = dict()
        
        self.encoder_aux_opt.zero_grad(set_to_none=True)
        self.obj_pos_opt.zero_grad(set_to_none=True)

        if step > self.aux_latency:
            fix_view_feat, fix_embed = self.encoder(fix_obs, return_infonce=True)
        else:
            fix_view_feat = self.encoder(fix_obs)
        pred_obj_pos = self.obj_pos_head(fix_view_feat)
        obj_pos_loss = F.mse_loss(pred_obj_pos, obj_pos)
        total_loss = obj_pos_loss 
        if self.use_tb:
            metrics['aux_obj_pos_loss'] = obj_pos_loss.item()
        if step > self.aux_latency:
            move_view_feat, move_embed = self.encoder(move_obs, return_infonce=True)
            
            contrastive_loss = info_nce_loss(move_embed, fix_embed, temperature=self.auxiliary_temp)
            aux_loss = contrastive_loss * self.aux_coef
            total_loss = total_loss + aux_loss 
            
            if self.use_tb:
                metrics['aux_contrastive_loss'] = contrastive_loss.item()
                metrics['aux_lr'] = self.encoder_aux_opt.param_groups[0]['lr']
        else:
            if self.use_tb:
                metrics['aux_contrastive_loss'] = 0
                metrics['aux_lr'] = self.encoder_aux_opt.param_groups[0]['lr']
        total_loss.backward()
        self.encoder_aux_opt.step()
        self.obj_pos_opt.step()
        if self.use_tb:
            grad = self.encoder.resnet18[0].weight.grad
            metrics['grad_aux_mean'] = grad.mean().item() if grad is not None else 0
            metrics['grad_aux_max'] = grad.max().item() if grad is not None else 0
            metrics['grad_aux_min'] = grad.min().item() if grad is not None else 0

        return metrics

    def update(self, replay_iter, trajs, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, reward, discount, next_obs, obj_pos = utils.to_torch(
            batch, self.device)

        # auxiliary
        l = obs.shape[1] // 2
        fix_obs=obs.float()[:, :l]
        move_obs=obs.float()[:, l:]
        fix_next_obs = next_obs.float()[:, :l]
        
        # augment
        obs = self.aug(fix_obs)
        original_obs = obs.clone()
        next_obs = self.aug(fix_next_obs)
        # import ipdb;ipdb.set_trace()
        original_move_obs = move_obs.clone()

        # TODO: not elegant
        # strong augmentation + SRM
        if l % 3 == 0:
            aug_obs = random_mask_freq_v2(random_overlay(original_obs))
            if step > self.aux_latency:
                aug_move_obs = random_mask_freq_v2(random_overlay(original_move_obs))
                aug_move_obs = self.encoder(aug_move_obs)  
            else:
                aug_move_obs = None
        else:
            aug_obs = random_mask_freq_v2(random_overlay(original_obs[:, :l-1]))
            # print("", aug_obs.shape, )
            aug_obs = torch.cat([aug_obs, original_obs[:, l-1:l]], dim=1)
            if step > self.aux_latency:
                aug_move_obs = random_mask_freq_v2(random_overlay(original_move_obs[:, :l-1]))
                aug_move_obs = torch.cat([aug_move_obs, original_move_obs[:, l-1:l]], dim=1)
                aug_move_obs = self.encoder(aug_move_obs)       
            else:
                aug_move_obs = None
            
        aug_obs = self.encoder(aug_obs)
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step, aug_obs, aug_move_obs))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update auxiliary task
        metrics.update(self.update_auxiliary(step, fix_obs, move_obs, obj_pos)) 
        
        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    
