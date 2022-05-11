import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import utils
import algorithms.modules as m


class SAC_FISHER(object):
        def __init__(self, obs_shape, action_shape, args):
                self.discount = args.discount
                self.critic_tau = args.critic_tau
                self.encoder_tau = args.encoder_tau
                self.actor_update_freq = args.actor_update_freq
                self.critic_target_update_freq = args.critic_target_update_freq
                self.f_reg = args.f_reg

                """
                shared_cnn = m.SharedCNN(obs_shape, args.num_shared_layers, args.num_filters).cuda()
                head_cnn = m.HeadCNN(shared_cnn.out_shape, args.num_head_layers, args.num_filters).cuda()
                actor_encoder = m.Encoder(
                        shared_cnn,
                        head_cnn,
                        m.RLProjection(head_cnn.out_shape, args.projection_dim)
                )
                critic_encoder = m.Encoder(
                        shared_cnn,
                        head_cnn,
                        m.RLProjection(head_cnn.out_shape, args.projection_dim)
                )
                """
                self.iters = args.iters
                actor_encoder = m.featEncoder( 
                            m.RLProjection(obs_shape, args.projection_dim)
                )
                critic_encoder = m.featEncoder(
                            m.RLProjection(obs_shape, args.projection_dim)
                )

                self.actor = m.Actor(actor_encoder, action_shape, args.hidden_dim, args.actor_log_std_min, args.actor_log_std_max).cuda()
                self.bp = deepcopy(self.actor)
                self.critic = m.Critic(critic_encoder, action_shape, args.hidden_dim).cuda()
                self.critic_target = deepcopy(self.critic)

                self.log_alpha = torch.tensor(np.log(args.init_temperature)).cuda()
                self.log_alpha.requires_grad = True
                self.target_entropy = -np.prod(action_shape)

                self.actor_optimizer = torch.optim.Adam(
                        self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
                )
                self.critic_optimizer = torch.optim.Adam(
                        self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
                )
                self.log_alpha_optimizer = torch.optim.Adam(
                        [self.log_alpha], lr=args.alpha_lr, betas=(args.alpha_beta, 0.999)
                )

                self.train()
                self.critic_target.train()

        def train(self, training=True):
                self.training = training
                self.actor.train(training)
                self.critic.train(training)

        def eval(self):
                self.train(False)

        @property
        def alpha(self):
                return self.log_alpha.exp()
                


        def dist_critic(self, states, actions, target=False, stop_gradient=False):
                if target:
                        q1, q2 = self.critic_target(states, actions)
                else:
                        q1, q2 = self.critic(states, actions)
                log_probs = self.bp.log_probs(states, actions)
                if stop_gradient:
                        #log_probs = tf.stop_gradient(log_probs)
                        log_probs = log_probs.detach()
                return (q1 + log_probs, q2 + log_probs)

        def grad_penalty(self, states, actions):
                q1, q2 = self.critic(state, actions)
                grad_1 = autograd.grad(
                        outputs=q1,
                        inputs=actions,
                        grad_outputs=torch.ones_like(q1),
                        create_graph=True,
                        retain_graph=True
                )[0]
                grad_2 = autograd.grad(
                        outputs=q2,
                        inputs=actions,
                        grad_outputs=torch.ones_like(q2),
                        create_graph=True,
                        retain_graph=True
                )[0]
                grad1 = grad1.view(q1.shape[0], -1)
                grad2 = grad2.view(q2.shape[0], -1)

                return (torch.square(grad1)+torch.square(grad2)).mean()

        def _obs_to_input(self, obs):
                if isinstance(obs, utils.LazyFrames):
                        _obs = np.array(obs)
                else:
                        _obs = obs
                _obs = torch.FloatTensor(_obs).cuda()
                _obs = _obs.unsqueeze(0)
                return _obs

        def select_action(self, obs):
                _obs = self._obs_to_input(obs)
                with torch.no_grad():
                        mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)
                return mu.cpu().data.numpy().flatten()

        def sample_action(self, obs):
                _obs = self._obs_to_input(obs)
                with torch.no_grad():
                        mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
                return pi.cpu().data.numpy().flatten()

        def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
                grad_loss = grad_penalty(obs, action)
                
                with torch.no_grad():
                        _, policy_action, log_pi, _ = self.actor(next_obs)
                        target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
                        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
                        target_Q = reward + (not_done * self.discount * target_V)

                current_Q1, current_Q2 = self.critic(obs, action)
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
                if L is not None:
                        L.log('train_critic/loss', critic_loss, step)

                loss = self.f_reg * grad_loss + critic_loss

                self.critic_optimizer.zero_grad()
                loss.backward()
                self.critic_optimizer.step()

        def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
                
                for i in range(self.iters):
                    _, pi, log_pi, log_std = self.actor(obs, detach=True)
                    #actor_Q1, actor_Q2 = self.critic(obs, pi, detach=True)

                    actor_Q1, actor_Q2 = self.dist_critic(obs, pi)
                    actor_Q = torch.min(actor_Q1, actor_Q2)
                    actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

                    if L is not None and i == self.iters-1:
                            L.log('train_actor/loss', actor_loss, step)
                            entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                                                                                    ) + log_std.sum(dim=-1)

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    if update_alpha and i == self.iters-1:
                            self.log_alpha_optimizer.zero_grad()
                            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()

                            if L is not None:
                                    L.log('train_alpha/loss', alpha_loss, step)
                                    L.log('train_alpha/value', self.alpha, step)

                            alpha_loss.backward()
                            self.log_alpha_optimizer.step()

        def soft_update_critic_target(self):
                utils.soft_update_params(
                        self.critic.Q1, self.critic_target.Q1, self.critic_tau
                )
                utils.soft_update_params(
                        self.critic.Q2, self.critic_target.Q2, self.critic_tau
                )
                utils.soft_update_params(
                        self.critic.encoder, self.critic_target.encoder,
                        self.encoder_tau
                )

        def update(self, replay_buffer, L, step):
                obs, action, reward, next_obs, not_done = replay_buffer.sample()

                self.update_critic(obs, action, reward, next_obs, not_done, L, step)

                if step % self.actor_update_freq == 0:
                        self.update_actor_and_alpha(obs, L, step)

                if step % self.critic_target_update_freq == 0:
                        self.soft_update_critic_target()
