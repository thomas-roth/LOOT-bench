import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from agent.config import QCFQLConfig
from networks.actor_vector_field import ActorVectorField
from networks.value import Value


class QCFQLAgent(nn.Module):
    """Port of `qc/agents/acfql.py::ACFQLAgent`.

    JAX's `flax.struct.PyTreeNode` agent is immutable -- every update returns
    a brand-new agent (`agent.replace(...)`) because `jit`/`grad` require
    pure functions with explicit state in/out. PyTorch's autograd is
    imperative by design, so this is an ordinary mutable `nn.Module`:
    `optimizer.step()` mutates parameters in place, and `update()` returns
    only the info dict, not a new agent.

    JAX threads an explicit `rng` key through every stochastic call (needed
    for reproducibility under its functional/parallel execution model); this
    port uses PyTorch's global RNG instead, except where `critic_loss`/
    `actor_loss`/`sample_actions` accept optional injected noise -- added
    purely so the gold-data test can feed identical draws to both sides
    (JAX's `jax.random` and `torch.randn` never produce the same numbers).
    These arguments are inert in real training: omitted, each method samples
    its own noise exactly as JAX does.

    The one piece that is NOT just an idiom swap: JAX's `actor_loss` scores
    the actor's action through the critic *without* `params=grad_params`
    (`qs = self.network.select('critic')(obs, actions=actor_actions)`), which
    -- per `TrainState.__call__`'s own docstring ("the default behavior is to
    stop the gradients... you need to explicitly provide the parameters to
    flow the gradients") -- means the critic's own weights get *zero*
    gradient from the actor's `q_loss`; only `actor_actions` (hence
    `actor_onestep_flow`) does. PyTorch's autograd has no per-call
    "use a frozen snapshot instead" switch, so this port explicitly toggles
    `requires_grad_(False)` on the critic's parameters around that one call
    -- the same pattern OpenAI SpinningUp's DDPG uses for exactly this
    situation. Without it, a single shared optimizer would incorrectly let
    the actor's objective update the critic's weights too.
    """

    def __init__(self, config: QCFQLConfig):
        super().__init__()
        assert config.ob_dims is not None and config.action_dim is not None, (
            "config.ob_dims/action_dim must be set (use QCFQLAgent.create())"
        )
        self.config = config

        obs_dim = config.ob_dims[-1]
        action_dim = config.action_dim
        full_action_dim = (
            action_dim * config.horizon_length if config.action_chunking else action_dim
        )

        self.actor_bc_flow = ActorVectorField(
            obs_dim=obs_dim,
            action_dim=full_action_dim,
            hidden_dims=config.actor_hidden_dims,
            layer_norm=config.actor_layer_norm,
            use_time=True,
            use_fourier_features=config.use_fourier_features,
            fourier_feature_dim=config.fourier_feature_dim,
        )
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
        self.critic_target = copy.deepcopy(self.critic)

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
        config.ob_dims = tuple(ex_observations.shape)
        config.action_dim = ex_actions.shape[-1]
        return cls(config)

    def _reshape_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Fold horizon_length into action_dim: (B, T, act_dim) -> (B, T*act_dim)."""
        if self.config.action_chunking:
            return actions.reshape(actions.shape[0], -1)
        return actions[..., 0, :]

    def critic_loss(
        self,
        batch: dict[str, torch.Tensor],
        next_action_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """FQL critic loss (TD regression against a frozen target)."""
        batch_actions = self._reshape_actions(batch["actions"])
        next_obs = batch["next_observations"][:, -1, :]

        with torch.no_grad():
            next_actions = self.sample_actions(next_obs, noise=next_action_noise)
            next_qs = self.critic_target(next_obs, next_actions)
            if self.config.q_agg == "min":
                next_q = next_qs.min(dim=0).values
            else:
                next_q = next_qs.mean(dim=0)

            reward = batch["rewards"][:, -1]
            mask = batch["masks"][:, -1]
            discount_h = self.config.discount**self.config.horizon_length
            target_q = reward + discount_h * mask * next_q

        q = self.critic(batch["observations"], batch_actions)
        valid = batch["valid"][:, -1]
        # `q` is (num_ensembles, batch), `target_q` is (batch,) -- each head
        # independently regresses toward the same target, matching JAX's
        # `jnp.square(q - target_q)` intentional broadcast. Not `F.mse_loss`
        # (same result, but warns about the shape mismatch).
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
        """FQL actor loss: flow-matching BC + optional DDPG distillation."""
        batch_actions = self._reshape_actions(batch["actions"])
        batch_size, full_action_dim = batch_actions.shape
        device = batch_actions.device

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
            with torch.no_grad():
                target_flow_actions = self.compute_flow_actions(
                    batch["observations"], noises
                )
            actor_actions = self.actor_onestep_flow(batch["observations"], noises)
            distill_loss = F.mse_loss(actor_actions, target_flow_actions)

            actor_actions_clipped = torch.clamp(actor_actions, -1.0, 1.0)
            # Freeze the critic's own weights so only `actor_actions` (hence
            # `actor_onestep_flow`) receives gradient from q_loss -- see the
            # class docstring for why this is required, not optional.
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
        """Single combined gradient step (matches JAX's one joint `apply_loss_fn`
        over `total_loss = critic_loss + actor_loss`, computed from the same
        pre-step snapshot for both losses)."""
        critic_loss, critic_info = self.critic_loss(batch)
        actor_loss, actor_info = self.actor_loss(batch)
        total_loss = critic_loss + actor_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        tau = self.config.tau
        with torch.no_grad():
            for p, tp in zip(
                self.critic.parameters(), self.critic_target.parameters(), strict=True
            ):
                tp.data.mul_(1 - tau).add_(p.data, alpha=tau)

        return {**critic_info, **actor_info}

    def batch_update(self, batch: dict[str, torch.Tensor]) -> dict:
        """`utd_ratio` sequential updates (JAX's `jax.lax.scan` -- a compiled
        loop, same sequence of updates as an eager Python loop)."""
        infos = []
        utd_ratio = next(iter(batch.values())).shape[0]
        for i in range(utd_ratio):
            mini_batch = {k: v[i] for k, v in batch.items()}
            infos.append(self.update(mini_batch))
        return {k: float(np.mean([info[k] for info in infos])) for k in infos[0]}

    def sample_actions(
        self, observations: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        assert self.config.action_dim is not None
        batch_size = observations.shape[0]
        full_action_dim = self.config.action_dim * (
            self.config.horizon_length if self.config.action_chunking else 1
        )

        if self.config.actor_type == "distill-ddpg":
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
            # NOTE: mirrors the source exactly -- `critic_loss` checks
            # `q_agg == "min"` (else mean), this branch checks `q_agg ==
            # "mean"` (else min). Opposite fallback defaults in the original;
            # both select mean for the actual config default ("mean").
            q = qs.mean(dim=0) if self.config.q_agg == "mean" else qs.min(dim=0).values
            indices = q.argmax(dim=-1)
            return actions[torch.arange(batch_size), indices]

        raise ValueError(f"Unknown actor_type: {self.config.actor_type}")

    @torch.no_grad()
    def compute_flow_actions(
        self, observations: torch.Tensor, noises: torch.Tensor
    ) -> torch.Tensor:
        """Euler-integrate the BC flow model from noise to an action."""
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
