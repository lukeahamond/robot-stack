"""Training stages, in the order they run:

    1. encoder      reconstruction + action prediction, then frozen (the shared latent space)
    2. world model  latent dynamics + irreversible-failure head, on all data
    3. planner      hindsight-relabelled subgoals: words and the latent at the subgoal's end
    4. action model behaviour cloning on successful expert data, plus a future-latent side objective
    5. critic       subgoal-done and value heads, on successes and failures

    corrections     the agent acts, the expert relabels its states, everything is refitted
    rl_finetune     deploy, log, refit the critic, advantage-weighted update of the action model
"""
import torch
import torch.nn.functional as F

from .data import CORRECTED, EXPERT, POLICY, SLOPPY, Buffer, seed_base
from .env import DONE, ToyKitchen
from .models import Models


def _fit(params, steps, lr, loss_fn):
    params = [p for p in params if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(steps, 1))
    tail = []
    for i in range(steps):
        loss = loss_fn()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if i >= steps - max(1, steps // 10):
            tail.append(loss.item())
    return sum(tail) / max(len(tail), 1)


def _accelerator(cfg):
    """The encoder is the only convolutional stage, so it alone is worth putting on a GPU."""
    if cfg.encoder_device != "auto":
        return cfg.encoder_device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_encoder(cfg, m, buf, steps=None):
    """Reconstruction on all frames plus action prediction on expert steps, then freeze."""
    frames, S, dev = buf.F["frames"], buf.S, _accelerator(cfg)
    for part in (m.encoder, m.decoder, m.aux):
        part.to(dev)
    pool = bc_probs(buf).nonzero().squeeze(1).cpu()
    half = cfg.batch // 2

    def loss():
        i = pool[torch.randint(len(pool), (half,))]
        fi = torch.cat([torch.randint(len(frames), (half,)), S["fi"][i].cpu()])
        x = frames[fi].to(dev).float() / 255
        z = m.encoder(x)
        rec = ((m.decoder(z) - x) ** 2 * (1 + 10 * x)).mean()  # upweight the small bright objects
        chunk = m.aux(z[half:], buf.F["prop"][fi[half:]].to(dev), S["sg"][i].to(dev))
        target = S["chunk"][i].to(dev)
        act = F.mse_loss(chunk[..., :2], target[..., :2]) \
            + F.binary_cross_entropy_with_logits(chunk[..., 2], target[..., 2])
        return cfg.w_reconstruct * rec + act

    params = [q for part in (m.encoder, m.decoder, m.aux) for q in part.parameters()]
    out = _fit(params, cfg.steps_encoder if steps is None else steps, cfg.lr, loss)
    buf.encode(m.encoder, cfg.device)  # while the encoder is still on the accelerator
    for part in (m.encoder, m.decoder, m.aux):
        part.to(cfg.device)
    for q in m.encoder.parameters():
        q.requires_grad_(False)
    return out


def reencode(cfg, m, buf):
    dev = _accelerator(cfg)
    buf.encode(m.encoder.to(dev), cfg.device)
    m.encoder.to(cfg.device)


def train_world(cfg, m, buf, steps=None):
    S, Z, prop = buf.S, buf.Z, buf.F["prop"]
    n = len(S["fi"])

    def loss():
        total = 0.0
        for member in m.world.members:  # a different batch per member keeps the ensemble diverse
            i = torch.randint(n, (cfg.batch,), device=Z.device)
            fi = S["fi"][i]
            z2, p2, logit = m.world.step(member, Z[fi], prop[fi], S["executed"][i], m.world.task(S["instr"][i]))
            total = total + F.mse_loss(z2, Z[fi + 1]) + F.mse_loss(p2, prop[fi + 1]) \
                + 5.0 * F.binary_cross_entropy_with_logits(logit, S["fail"][i].float())
        return total

    return _fit(m.world.parameters(), cfg.steps_world if steps is None else steps, cfg.lr, loss)


def train_planner(cfg, m, buf, steps=None):
    Fd, Z = buf.F, buf.Z
    pool = (Fd["mode"] != POLICY).nonzero().squeeze(1)  # frames whose subgoal label is the oracle's

    def loss():
        f = pool[torch.randint(len(pool), (cfg.batch,), device=Z.device)]
        sg = Fd["sg"][f]
        verb, color, goal, _ = m.planner(Z[f], Fd["prop"][f], Fd["instr"][f], sg)
        active = (sg[:, 0] != DONE).float()
        out = F.cross_entropy(verb, sg[:, 0])
        out = out + (F.cross_entropy(color, sg[:, 1], reduction="none") * active).sum() / active.sum().clamp(min=1)
        valid = Fd["goal_valid"][f].float()
        err = ((goal - Z[Fd["goal_fi"][f]]) ** 2).mean(-1)
        return out + (err * valid).sum() / valid.sum().clamp(min=1)

    return _fit(m.planner.parameters(), cfg.steps_planner if steps is None else steps, cfg.lr, loss)


def train_action(cfg, m, buf, probs, steps=None):
    """probs: sampling weight per step. Uniform over expert successes = behaviour cloning;
    exp(advantage / beta) = advantage-weighted regression."""
    S, Z, prop = buf.S, buf.Z, buf.F["prop"]
    B = cfg.batch

    def loss():
        i = torch.multinomial(probs, B, replacement=True)
        fi = S["fi"][i]
        # Half the time condition on the planner's predicted latent, as at runtime;
        # otherwise on the hindsight latent, which is only meaningful if the subgoal was reached.
        with torch.no_grad():
            _, _, predicted, _ = m.planner(Z[fi], prop[fi], S["instr"][i], S["sg"][i])
        from_planner = torch.rand(B, 1, device=Z.device) < 0.5
        goal = torch.where(from_planner, predicted, Z[S["goal_fi"][i]])
        available = from_planner | S["goal_valid"][i][:, None]
        use_latent = ((torch.rand(B, 1, device=Z.device) > cfg.p_drop_latent) & available).float()
        use_words = (torch.rand(B, 1, device=Z.device) > cfg.p_drop_words).float()
        use_words = torch.maximum(use_words, 1 - use_latent)  # never drop both channels
        chunk, future = m.action(Z[fi], prop[fi], S["sg"][i], goal, use_words, use_latent)
        target = S["chunk"][i]
        return F.mse_loss(chunk[..., :2], target[..., :2]) \
            + F.binary_cross_entropy_with_logits(chunk[..., 2], target[..., 2]) \
            + 0.5 * F.mse_loss(future, Z[S["future_fi"][i]])

    return _fit(m.action.parameters(), cfg.steps_action if steps is None else steps, cfg.lr, loss)


def train_critic(cfg, m, buf, steps=None):
    Fd, Z = buf.F, buf.Z
    n, B = len(Z), cfg.batch

    def loss():
        f = torch.randint(n, (B,), device=Z.device)
        rand = torch.stack([torch.randint(2, (B,), device=Z.device), torch.randint(3, (B,), device=Z.device)], -1)
        # Query the subgoal that was in force when this frame was reached (what the runtime
        # asks), the frame's own subgoal, or a random one.
        r = torch.rand(B, 1, device=Z.device)
        q = torch.where(r < 1 / 3, Fd["prev_sg"][f], torch.where(r < 2 / 3, Fd["sg"][f], rand))
        q = torch.where(q[:, :1] == DONE, rand, q)
        label = Fd["ach"][f, q[:, 0], q[:, 1]].float()
        return F.binary_cross_entropy_with_logits(m.critic.done_logit(Z[f], Fd["prop"][f], q), label) \
            + F.binary_cross_entropy_with_logits(m.critic.value_logit(Z[f], Fd["prop"][f], Fd["instr"][f]), Fd["ret"][f])

    return _fit(m.critic.parameters(), cfg.steps_critic if steps is None else steps, cfg.lr, loss)


def bc_probs(buf):
    """Steps whose label is an expert action: successful demonstrations and all corrected steps."""
    S = buf.S
    return (((S["mode"] == EXPERT) & S["success"]) | (S["mode"] == CORRECTED)).float()


def train_all(cfg, eps, log=print):
    torch.manual_seed(cfg.seed)
    m = Models(cfg).to(cfg.device)
    buf = Buffer(cfg, eps).build(cfg.device)
    log(f"encoder      loss {train_encoder(cfg, m, buf):.5f}")
    log(f"world model  loss {train_world(cfg, m, buf):.4f}")
    log(f"planner      loss {train_planner(cfg, m, buf):.4f}")
    log(f"action model loss {train_action(cfg, m, buf, bc_probs(buf)):.4f}")
    log(f"critic       loss {train_critic(cfg, m, buf):.4f}")
    corrections(cfg, m, buf, log)
    return m, buf


def corrections(cfg, m, buf, log=print):
    """Let the agent act, have the expert relabel every state it visits, retrain. Repeat.

    Plain imitation only sees states the expert visits, so a small error carries the agent
    somewhere it has no data for and it never recovers. This is the standard remedy
    (DAgger). On a real robot the labels come from a person taking over when the robot
    goes wrong.
    """
    from .runtime import Agent

    for r in range(cfg.correction_rounds):
        agent = Agent(cfg, m, rollout_check=False, seed=cfg.seed + 100 + r)
        eps = []
        for k in range(cfg.correction_episodes):
            env = ToyKitchen(cfg, seed_base(cfg) + 1_000_000 + r * 100_000 + k)
            eps.append(agent.run(env, corrections=True)[1])
        buf.add(eps)
        reencode(cfg, m, buf.build(cfg.device))
        train_world(cfg, m, buf, cfg.steps_refit)
        train_planner(cfg, m, buf, cfg.steps_refit)
        train_action(cfg, m, buf, bc_probs(buf), cfg.steps_refit)
        train_critic(cfg, m, buf, cfg.steps_refit)
        rate = sum(e["success"] for e in eps) / len(eps)
        log(f"correction round {r + 1}/{cfg.correction_rounds}: {len(eps)} episodes relabelled, "
            f"agent success before retraining {rate:.2f}")


def rl_finetune(cfg, m, buf, log=print):
    """Deploy -> log -> refit critic -> advantage-weighted update of the action model.

    The policy gradient never passes through the world model: the value function is fitted
    to real outcomes and is used only to weight which logged actions to imitate.
    """
    from .runtime import Agent  # local import: runtime depends on this module's siblings

    for r in range(cfg.rl_rounds):
        agent = Agent(cfg, m, explore=cfg.rl_explore, seed=cfg.seed + r)
        eps = []
        for k in range(cfg.rl_episodes):
            env = ToyKitchen(cfg, seed_base(cfg) + 2_000_000 + r * 100_000 + k)
            eps.append(agent.run(env, record=True)[1])
        buf.add(eps)
        reencode(cfg, m, buf.build(cfg.device))
        train_critic(cfg, m, buf, cfg.rl_steps_critic)
        train_world(cfg, m, buf, cfg.rl_steps_world)
        S = buf.S
        with torch.no_grad():
            V = torch.sigmoid(m.critic.value_logit(buf.Z[S["fi"]], buf.F["prop"][S["fi"]], S["instr"]))
            w = torch.exp((S["ret"] - V) / cfg.rl_beta).clamp(max=20.0)
            # A failed episode has a return of 0 and the critic learns a value near 0 on its states, so
            # its advantage is near 0 and its weight near 1, the same as a good step. Sloppy steps are
            # labelled with deliberately wrong actions. Neither should be imitated. Corrected steps carry
            # the expert's label whatever the outcome, so they are kept.
            keep = (S["mode"] == CORRECTED) | ((S["mode"] != SLOPPY) & S["success"])
            w = w * keep.float()
        if not w.sum() > 0:
            log(f"rl round {r + 1}/{cfg.rl_rounds}: no step worth imitating, action model left as it was")
            continue
        train_action(cfg, m, buf, w, cfg.rl_steps_action)
        rate = sum(e["success"] for e in eps) / len(eps)
        log(f"rl round {r + 1}/{cfg.rl_rounds}: {len(eps)} episodes logged, success while exploring {rate:.2f}")
    return m
