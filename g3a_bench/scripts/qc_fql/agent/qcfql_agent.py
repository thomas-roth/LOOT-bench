"""The QC-FQL agent: losses, joint update, and action sampling.

Port of `qc/agents/acfql.py::ACFQLAgent`. The algorithm in one paragraph:
treat a length-H action chunk as one flat macro-action; train a flow-matching
model (`actor_bc_flow`) to imitate demonstrated chunks; train a critic
ensemble on H-step TD backups over those chunks; and (in the default
"distill-ddpg" mode) distill the multi-step flow into a one-shot policy
(`actor_onestep_flow`) that is additionally pushed uphill on the critic --
BC keeps it near the data, Q makes it better than the data.

Structural translation from the reference, and why each move is safe:

- `flax.struct.PyTreeNode` + `agent.replace(...)` (immutability required by
  jit/grad purity) -> ordinary mutable `nn.Module`; `update()` mutates in
  place and returns just the info dict.
- One `jax.grad` over `total_loss = critic + actor` with a single optax
  optimizer -> one `backward()` on the summed loss with a single torch
  optimizer over all three networks. Both losses see the same pre-step
  parameter snapshot, exactly as under the reference's single grad call.
- Explicit rng threading -> torch's global RNG. The optional noise
  arguments on `critic_loss`/`actor_loss`/`sample_actions` exist ONLY so
  gold tests can inject the reference's exact jax.random draws (the two
  frameworks' RNGs can never agree); omitted in real training, each method
  samples its own noise like the reference does.
- `jax.lax.scan` in `batch_update` -> a Python loop (scan is a compilation
  detail, the update sequence is identical).

The one subtle, load-bearing piece: gradient isolation in `actor_loss`.
The reference scores the actor's action through the critic WITHOUT passing
`params=grad_params` -- under `TrainState.__call__`'s convention that means
the critic acts as a frozen scorer: zero gradient to its own weights, while
gradient still flows through the action into `actor_onestep_flow`. torch
has no per-call frozen-view switch, so this port toggles
`requires_grad_(False/True)` around that single critic call. requires_grad
is captured at graph-construction (forward) time, so re-enabling it
immediately afterwards does not retroactively add the edges back -- the
later combined backward() stays clean. Removing the toggle makes
`test_critic_only_gets_gradient_from_its_own_loss` fail (verified), and the
failure is invisible to every loss-value test since forward values are
unchanged -- don't "simplify" this away.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from g3a_bench.scripts.qc_fql.agent.config import QCFQLConfig
from g3a_bench.scripts.qc_fql.networks.actor_vector_field import ActorVectorField
from g3a_bench.scripts.qc_fql.networks.value import Value


class QCFQLAgent(nn.Module):
    def __init__(self, config: QCFQLConfig):
        super().__init__()
        assert config.ob_dims is not None and config.action_dim is not None, (
            "config.ob_dims/action_dim must be set (use QCFQLAgent.create())"
        )
        self.config = config

        obs_dim = config.ob_dims[-1]
        action_dim = config.action_dim
        # Chunking folds the horizon into the action width: every network
        # sees (obs, H*act_dim) and is oblivious to the chunk structure.
        full_action_dim = (
            action_dim * config.horizon_length if config.action_chunking else action_dim
        )

        # The flow field proper: conditioned on denoising time.
        self.actor_bc_flow = ActorVectorField(
            obs_dim=obs_dim,
            action_dim=full_action_dim,
            hidden_dims=config.actor_hidden_dims,
            layer_norm=config.actor_layer_norm,
            use_time=True,
            use_fourier_features=config.use_fourier_features,
            fourier_feature_dim=config.fourier_feature_dim,
        )
        # The distilled one-shot policy: noise -> action, no time input
        # (the reference builds it without fourier features for the same reason).
        self.actor_onestep_flow = ActorVectorField(
            obs_dim=obs_dim,
            action_dim=full_action_dim,
            hidden_dims=config.actor_hidden_dims,
            layer_norm=config.actor_layer_norm,
            use_time=False,
        )
        self.critic = Value(
            in_features=obs_dim + full_action_dim,
            hidden_dims=config.value_hidden_dims,
            layer_norm=config.layer_norm,
            num_ensembles=config.num_qs,
            use_actions=True,
        )
        # deepcopy reproduces the reference's params[target] = params[critic]
        # initialization; only Polyak updates touch it afterwards.
        self.critic_target = copy.deepcopy(self.critic)

        # ONE optimizer over all trainable networks, like the reference's
        # single optax transform over its whole ModuleDict. critic_target is
        # deliberately not in it.
        trainable_params = (
            list(self.critic.parameters())
            + list(self.actor_bc_flow.parameters())
            + list(self.actor_onestep_flow.parameters())
        )
        if config.weight_decay > 0:
            self.optimizer = torch.optim.AdamW(
                trainable_params, lr=config.lr, weight_decay=config.weight_decay
            )
        else:
            self.optimizer = torch.optim.Adam(trainable_params, lr=config.lr)

    @classmethod
    def create(
        cls, config: QCFQLConfig, ex_observations: np.ndarray, ex_actions: np.ndarray
    ) -> "QCFQLAgent":
        """Infer obs/action dims from example data, as the reference's create() does.

        `ex_observations` is one UNBATCHED example, shape (obs_dim,) -- the
        reference derives batch-shape handling from `len(ob_dims)`, so passing
        a batched (1, obs_dim) example corrupts that bookkeeping downstream.
        """
        config.ob_dims = tuple(ex_observations.shape)
        config.action_dim = ex_actions.shape[-1]
        return cls(config)

    def _reshape_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """(B, T, act_dim) -> (B, T*act_dim) chunk flattening; without
        chunking, keep only the first step (plain n-step SARSA-style target)."""
        if self.config.action_chunking:
            return actions.reshape(actions.shape[0], -1)
        return actions[..., 0, :]

    def critic_loss(
        self,
        batch: dict[str, torch.Tensor],
        next_action_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """H-step TD regression: Q(s_0, chunk) -> R_{0..H-1} + gamma^H * Q'(s_H, chunk').

        The batch comes from `Dataset.sample_sequence`, so `rewards[:, -1]`
        is already the discounted H-step sum and `next_observations[:, -1]`
        is s_H -- no reward math happens here.
        """
        batch_actions = self._reshape_actions(batch["actions"])
        next_obs = batch["next_observations"][:, -1, :]

        # Everything on the target side of the Bellman equation is
        # gradient-free, including the sampled next chunk (the reference's
        # sample_actions there uses stored params too).
        with torch.no_grad():
            next_actions = self.sample_actions(next_obs, noise=next_action_noise)
            next_qs = self.critic_target(next_obs, next_actions)
            if self.config.q_agg == "min":
                next_q = next_qs.min(dim=0).values
            else:
                next_q = next_qs.mean(dim=0)

            reward = batch["rewards"][:, -1]
            mask = batch["masks"][:, -1]
            # gamma^H, not gamma: the chunk spans H env steps.
            discount_h = self.config.discount**self.config.horizon_length
            target_q = reward + discount_h * mask * next_q

        q = self.critic(batch["observations"], batch_actions)
        # valid[:, -1] zeroes windows whose episode ended mid-chunk.
        valid = batch["valid"][:, -1]
        # (num_ensembles, B) - (B,) broadcast is intentional (each head
        # regresses to the same target, as in the reference's jnp.square);
        # F.mse_loss computes the same thing but warns about the shapes.
        critic_loss = ((q - target_q) ** 2 * valid).mean()

        info = {
            "critic_loss": critic_loss.item(),
            "q_mean": q.mean().item(),
            "q_max": q.max().item(),
            "q_min": q.min().item(),
        }
        return critic_loss, info

    def actor_loss(
        self,
        batch: dict[str, torch.Tensor],
        x0: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        distill_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Flow-matching BC + (distill-ddpg only) distillation and Q terms."""
        batch_actions = self._reshape_actions(batch["actions"])
        batch_size, full_action_dim = batch_actions.shape
        device = batch_actions.device

        # Linear-interpolation flow matching: x_t on the noise->data line,
        # regression target is the constant velocity x_1 - x_0.
        x_0 = (
            x0
            if x0 is not None
            else torch.randn(batch_size, full_action_dim, device=device)
        )
        x_1 = batch_actions
        t_ = t if t is not None else torch.rand(batch_size, 1, device=device)
        x_t = (1 - t_) * x_0 + t_ * x_1
        vel = x_1 - x_0

        pred = self.actor_bc_flow(batch["observations"], x_t, t_)

        if self.config.action_chunking:
            assert self.config.action_dim is not None
            # Unfold back to (B, T, act) so per-step `valid` can mask out
            # chunk positions that fall past an episode boundary.
            pred_r = pred.reshape(
                batch_size, self.config.horizon_length, self.config.action_dim
            )
            vel_r = vel.reshape(
                batch_size, self.config.horizon_length, self.config.action_dim
            )
            bc_flow_loss = ((pred_r - vel_r) ** 2 * batch["valid"][..., None]).mean()
        else:
            bc_flow_loss = F.mse_loss(pred, vel)

        distill_loss = torch.zeros((), device=device)
        q_loss = torch.zeros((), device=device)

        if self.config.actor_type == "distill-ddpg":
            noises = (
                distill_noise
                if distill_noise is not None
                else torch.randn(batch_size, full_action_dim, device=device)
            )
            # Distillation target: what the full Euler integration would do
            # from this exact noise. no_grad matches the reference (its
            # compute_flow_actions call also uses stored params only).
            with torch.no_grad():
                target_flow_actions = self.compute_flow_actions(
                    batch["observations"], noises
                )
            actor_actions = self.actor_onestep_flow(batch["observations"], noises)
            distill_loss = F.mse_loss(actor_actions, target_flow_actions)

            actor_actions_clipped = torch.clamp(actor_actions, -1.0, 1.0)
            # Gradient isolation (see module docstring): critic scores the
            # action as a frozen judge; only actor_onestep_flow may learn
            # from q_loss. Load-bearing -- removal is caught by a dedicated
            # gradient test but by no loss-value test.
            for p in self.critic.parameters():
                p.requires_grad_(False)
            qs = self.critic(batch["observations"], actor_actions_clipped)
            for p in self.critic.parameters():
                p.requires_grad_(True)
            q = qs.mean(dim=0)
            q_loss = -q.mean()

        actor_loss = bc_flow_loss + self.config.alpha * distill_loss + q_loss

        info = {
            "actor_loss": actor_loss.item(),
            "bc_flow_loss": bc_flow_loss.item(),
            "distill_loss": distill_loss.item()
            if torch.is_tensor(distill_loss)
            else distill_loss,
        }
        return actor_loss, info

    def update(self, batch: dict[str, torch.Tensor]) -> dict:
        """One joint gradient step + Polyak target update, mirroring the
        reference's single apply_loss_fn over critic_loss + actor_loss."""
        critic_loss, critic_info = self.critic_loss(batch)
        actor_loss, actor_info = self.actor_loss(batch)
        total_loss = critic_loss + actor_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # target <- tau * critic + (1 - tau) * target, critic only (the
        # actors have no targets in this algorithm).
        tau = self.config.tau
        with torch.no_grad():
            for p, tp in zip(
                self.critic.parameters(), self.critic_target.parameters(), strict=True
            ):
                tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

        return {**critic_info, **actor_info}

    def batch_update(self, batch: dict[str, torch.Tensor]) -> dict:
        """Run `update` once per leading-axis slice (the utd_ratio axis the
        training loop reshapes in); the reference does this via lax.scan."""
        infos = []
        utd_ratio = next(iter(batch.values())).shape[0]
        for i in range(utd_ratio):
            mini_batch = {k: v[i] for k, v in batch.items()}
            infos.append(self.update(mini_batch))
        return {k: float(np.mean([info[k] for info in infos])) for k in infos[0]}

    def sample_actions(
        self, observations: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Draw one action chunk per observation, shape (B, H*act_dim), in [-1, 1]."""
        assert self.config.action_dim is not None
        batch_size = observations.shape[0]
        full_action_dim = self.config.action_dim * (
            self.config.horizon_length if self.config.action_chunking else 1
        )

        if self.config.actor_type == "distill-ddpg":
            # One-shot: the distilled policy maps noise straight to a chunk.
            noises = (
                noise
                if noise is not None
                else torch.randn(
                    batch_size, full_action_dim, device=observations.device
                )
            )
            actions = self.actor_onestep_flow(observations, noises)
            return torch.clamp(actions, -1.0, 1.0)

        elif self.config.actor_type == "best-of-n":
            # Integrate the flow from N noise seeds, keep the chunk the
            # critic likes best.
            noises = (
                noise
                if noise is not None
                else torch.randn(
                    batch_size,
                    self.config.actor_num_samples,
                    full_action_dim,
                    device=observations.device,
                )
            )
            obs_expanded = observations.unsqueeze(1).expand(
                -1, self.config.actor_num_samples, -1
            )
            actions = self.compute_flow_actions(obs_expanded, noises)
            actions = torch.clamp(actions, -1.0, 1.0)

            qs = self.critic(obs_expanded, actions)
            # Reference quirk kept verbatim: critic_loss branches on
            # q_agg == "min" (else mean), this branch on q_agg == "mean"
            # (else min) -- opposite fallbacks, same result for both
            # documented values of q_agg.
            q = qs.mean(dim=0) if self.config.q_agg == "mean" else qs.min(dim=0).values
            indices = q.argmax(dim=-1)
            return actions[torch.arange(batch_size), indices]

        raise ValueError(f"Unknown actor_type: {self.config.actor_type}")

    @torch.no_grad()
    def compute_flow_actions(
        self, observations: torch.Tensor, noises: torch.Tensor
    ) -> torch.Tensor:
        """Euler-integrate the learned field from noise to an action chunk.

        Fixed step count, times sampled at i/flow_steps (left endpoints, so t
        never reaches 1.0), final result clipped -- all exactly as in the
        reference. no_grad also matches it: every reference call site uses
        stored params, so this is never differentiated through.
        """
        actions = noises
        for i in range(self.config.flow_steps):
            t = torch.full(
                (*observations.shape[:-1], 1),
                i / self.config.flow_steps,
                device=observations.device,
                dtype=observations.dtype,
            )
            vels = self.actor_bc_flow(observations, actions, t)
            actions = actions + vels / self.config.flow_steps
        return torch.clamp(actions, -1.0, 1.0)
