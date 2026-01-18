"""The learned parts of the stack. Each class is small on purpose and marks a seam:

    Encoder      -> a pretrained video encoder
    Planner      -> a vision-language model emitting a text subgoal and a latent subgoal
    ActionModel  -> a vision-language-action model emitting action chunks
    Critic       -> a value / success model on real frames
    WorldModel   -> a learned dynamics model, here in latent space
"""
import torch
import torch.nn as nn

from .env import DONE, PROP


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class Encoder(nn.Module):
    """Camera frame -> latent. Trained once (reconstruction + action prediction), then frozen: the shared space.

    The latent is [global features, keypoints]. Keypoints come from a spatial softmax over
    half-resolution feature maps, which gives sub-cell object positions and generalises to
    unseen positions far better than a flattened feature map does.
    """

    def __init__(self, cfg, keypoints=8):
        super().__init__()
        n = cfg.img
        c = (torch.arange(n) + 0.5) / n
        gy, gx = torch.meshgrid(c, c, indexing="ij")
        self.register_buffer("coords", torch.stack([gx, gy])[None])
        self.stem = nn.Sequential(nn.Conv2d(5, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())
        half = (torch.arange(n // 2) + 0.5) / (n // 2)
        hy, hx = torch.meshgrid(half, half, indexing="ij")
        self.register_buffer("kp_coords", torch.stack([hx, hy]).flatten(1))
        self.heat = nn.Conv2d(32, keypoints, 1)
        self.down = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(64 * (n // 8) ** 2, cfg.latent - 2 * keypoints), nn.LayerNorm(cfg.latent - 2 * keypoints),
        )

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255
        h = self.stem(torch.cat([x, self.coords.expand(len(x), -1, -1, -1)], 1))
        p = torch.softmax(self.heat(h).flatten(2), -1)          # (B, K, cells)
        xy = torch.einsum("bkn,cn->bkc", p, self.kp_coords)     # (B, K, 2), in [0, 1]
        return torch.cat([self.down(h), (xy.flatten(1) - 0.5) * 4], -1)


class Decoder(nn.Module):
    """Only used to train the encoder, and to look at what a latent means."""

    def __init__(self, cfg):
        super().__init__()
        s = cfg.img // 8
        self.net = nn.Sequential(
            nn.Linear(cfg.latent, 64 * s * s), nn.ReLU(), nn.Unflatten(1, (64, s, s)),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(), nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class TaskEmbed(nn.Module):
    """Instruction tokens -> vector. Stands in for a language encoder."""
    dim = 64

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(4, 16)

    def forward(self, instr):
        return self.emb(instr).flatten(1)


class SubgoalEmbed(nn.Module):
    """The 'words' channel: a (verb, colour) subgoal."""
    dim = 32

    def __init__(self):
        super().__init__()
        self.verb, self.color = nn.Embedding(3, 16), nn.Embedding(3, 16)

    def forward(self, sg):
        return torch.cat([self.verb(sg[:, 0]), self.color(sg[:, 1])], -1)


class Planner(nn.Module):
    """Slow model. Emits the next subgoal in words and as a predicted future latent."""

    def __init__(self, cfg):
        super().__init__()
        L, h = cfg.latent, cfg.hidden
        self.task, self.sg = TaskEmbed(), SubgoalEmbed()
        self.trunk = nn.Sequential(nn.Linear(L + PROP + TaskEmbed.dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.verb, self.color = nn.Linear(h, 3), nn.Linear(h, 3)
        self.goal = mlp(h + SubgoalEmbed.dim, h, L)

    def forward(self, z, prop, instr, sg=None):
        """With sg given (training) the latent goal is teacher-forced on the true subgoal."""
        h = self.trunk(torch.cat([z, prop, self.task(instr)], -1))
        verb, color = self.verb(h), self.color(h)
        if sg is None:
            sg = torch.stack([verb.argmax(-1), color.argmax(-1)], -1)
        return verb, color, self.goal(torch.cat([h, self.sg(sg)], -1)), sg

    @torch.no_grad()
    def plan(self, z, prop, instr):
        _, _, goal, sg = self(z, prop, instr)
        verb, color = int(sg[0, 0]), int(sg[0, 1])
        return (verb, 0 if verb == DONE else color), goal


class ActionModel(nn.Module):
    """Fast model. (frame latent, proprio, words, latent subgoal) -> a chunk of actions.

    Each action is [dx, dy, grip logit], with dx, dy relative to the gripper at the start
    of the chunk, in units of max_step. A second head predicts the latent `chunk` steps
    ahead: future prediction as a side objective, with no cost at runtime.
    """

    def __init__(self, cfg):
        super().__init__()
        L, h = cfg.latent, cfg.hidden
        self.H = cfg.chunk
        self.sg = SubgoalEmbed()
        self.trunk = nn.Sequential(nn.Linear(L + PROP + SubgoalEmbed.dim + 1 + L + 1, h), nn.ReLU(),
                                   nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.chunk, self.future = nn.Linear(h, self.H * 3), nn.Linear(h, L)

    def forward(self, z, prop, sg, goal, use_words, use_latent):
        """use_words / use_latent: float masks of shape (B, 1) that switch a channel off."""
        x = torch.cat([z, prop, self.sg(sg) * use_words, use_words, goal * use_latent, use_latent], -1)
        h = self.trunk(x)
        return self.chunk(h).view(-1, self.H, 3), self.future(h)


class Critic(nn.Module):
    """Reads real frames. done: is this subgoal achieved in this frame? value: will the task succeed?"""

    def __init__(self, cfg):
        super().__init__()
        L, h = cfg.latent, cfg.hidden
        self.task, self.sg = TaskEmbed(), SubgoalEmbed()
        self.done_net = mlp(L + PROP + SubgoalEmbed.dim, h, 1)
        self.value_net = mlp(L + PROP + TaskEmbed.dim, h, 1)

    def done_logit(self, z, prop, sg):
        return self.done_net(torch.cat([z, prop, self.sg(sg)], -1)).squeeze(-1)

    def value_logit(self, z, prop, instr):
        return self.value_net(torch.cat([z, prop, self.task(instr)], -1)).squeeze(-1)


class WorldModel(nn.Module):
    """Ensemble of latent dynamics models with an irreversible-failure head.

    Each member predicts the next latent, the next proprio and the chance that this action
    causes an irreversible failure. Members disagree where the data was thin, and that
    disagreement is the uncertainty signal.
    """

    def __init__(self, cfg):
        super().__init__()
        self.L, self.max_step = cfg.latent, cfg.max_step
        self.task = TaskEmbed()
        self.members = nn.ModuleList(mlp(cfg.latent + PROP + 3 + TaskEmbed.dim, cfg.hidden, cfg.latent + PROP + 1)
                                     for _ in range(cfg.ensemble))

    def step(self, member, z, prop, act, task_emb):
        out = member(torch.cat([z, prop, act, task_emb], -1))
        return z + out[:, : self.L], prop + out[:, self.L: self.L + PROP], out[:, -1]

    @torch.no_grad()
    def rollout_fail(self, z, prop, waypoints, grips, instr):
        """Imagine executing K actions. Returns each member's worst failure probability, shape (M,).

        z (1, L), prop (1, PROP), waypoints (K, 2) absolute, grips (K,) bool.
        """
        e = self.task(instr)
        worst = []
        for m in self.members:
            zm, pm, w = z, prop, z.new_zeros(())
            for k in range(len(waypoints)):
                g = grips[k].float().view(1, 1)
                act = torch.cat([(waypoints[k][None] - pm[:, :2]) / self.max_step, g], -1)
                zm, pm, logit = self.step(m, zm, pm, act, e)
                pm = torch.cat([pm[:, :2], g, pm[:, 3:]], -1)  # the grip command is known
                w = torch.maximum(w, torch.sigmoid(logit)[0])
            worst.append(w)
        return torch.stack(worst)


class AuxActionHead(nn.Module):
    """Throwaway action head used only while the encoder trains.

    Reconstruction alone gives latents that locate objects to within a few pixels, which is
    too coarse to grasp. Asking the latent to also predict expert actions makes it precise
    where control needs it.
    """

    def __init__(self, cfg):
        super().__init__()
        self.H = cfg.chunk
        self.sg = SubgoalEmbed()
        self.net = mlp(cfg.latent + PROP + SubgoalEmbed.dim, cfg.hidden, cfg.chunk * 3)

    def forward(self, z, prop, sg):
        return self.net(torch.cat([z, prop, self.sg(sg)], -1)).view(-1, self.H, 3)


class Models(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder, self.decoder, self.aux = Encoder(cfg), Decoder(cfg), AuxActionHead(cfg)
        self.planner, self.action = Planner(cfg), ActionModel(cfg)
        self.critic, self.world = Critic(cfg), WorldModel(cfg)
