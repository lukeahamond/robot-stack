"""The runtime loop and the ablation harness.

Three rates:
    planner       fires on events only (new task, subgoal done, regression, timeout, repeated veto)
    action model  once per `exec_h` actions, emitting a chunk
    controller    `substeps` ticks per action
"""
import numpy as np
import torch

from .controller import Controller
from .data import CORRECTED, POLICY, Recorder
from .env import DONE, ToyKitchen
from .expert import expert_action, oracle_subgoal


class Agent:
    def __init__(self, cfg, models, critic=True, rollout_check=True, words=True, latent=True,
                 timer=None, explore=0.0, seed=0):
        self.cfg, self.m = cfg, models.eval()
        self.critic, self.rollout_check = critic, rollout_check
        self.words, self.latent = float(words), float(latent)
        self.timer = cfg.timer_chunks if timer is None else timer
        self.explore = explore
        self.rng = np.random.default_rng(seed)
        self.dev = next(models.parameters()).device

    def _see(self, env):
        frame = torch.from_numpy(env.render())[None].to(self.dev)
        return self.m.encoder(frame), torch.from_numpy(env.proprio())[None].to(self.dev)

    def _done_prob(self, z, prop, sg):
        sg_t = torch.tensor([sg], device=self.dev)
        return torch.sigmoid(self.m.critic.done_logit(z, prop, sg_t)).item()

    @torch.no_grad()
    def run(self, env, record=False, corrections=False):
        """Runs one episode. Returns (stats, episode record or None).

        corrections=True records what the expert would have done at every state the agent
        visits, and the correct subgoal, as imitation labels. The agent's own actions are
        still the ones executed.
        """
        c, m = self.cfg, self.m
        ctrl = Controller(c)
        rec = Recorder(env, CORRECTED if corrections else POLICY) if record or corrections else None
        instr = torch.from_numpy(env.instruction)[None].to(self.dev)
        ones = torch.ones(1, 1, device=self.dev)
        sg, prev, goal = None, None, None
        need_plan, by_done, chunks, vetoes, t = True, False, 0, 0, 0
        stats = dict(planner_calls=0, vetoes=0, stopped_early=False)
        z, prop = self._see(env)

        while not env.done and t < c.max_actions:
            # ---- slow: planner, on events only ----
            if need_plan:
                new, goal = m.planner.plan(z, prop, instr)
                stats["planner_calls"] += 1
                if new != sg:
                    # Remember the finished subgoal only if the critic confirmed it, so a
                    # later regression (a slip) can be detected.
                    prev = sg if by_done else None
                    sg = new
                need_plan, by_done, chunks, vetoes = False, False, 0, 0
                if sg[0] == DONE:
                    stats["stopped_early"] = not env.done   # the planner said done before the task was
                    break

            # ---- medium: action model emits a chunk ----
            sg_t = torch.tensor([sg], device=self.dev)
            chunk, _ = m.action(z, prop, sg_t, goal, ones * self.words, ones * self.latent)
            rel = chunk[0, : c.exec_h, :2].cpu().numpy()
            grip = (chunk[0, : c.exec_h, 2] > 0).cpu().numpy()
            p0 = env.grip_xy.copy()
            waypoints = np.clip(p0 + rel * c.max_step, 0.0, 1.0)

            # ---- rollout check: only when the chunk would release a held object ----
            if self.rollout_check and prop[0, 3] > 0.5 and not grip.all():
                risk = m.world.rollout_fail(z, prop, torch.from_numpy(waypoints).float().to(self.dev),
                                            torch.from_numpy(grip).to(self.dev), instr)
                if risk.mean().item() > c.veto_fail_prob or risk.std().item() > c.veto_disagreement:
                    grip[:] = True  # hold on and let the action model keep adjusting
                    vetoes += 1
                    stats["vetoes"] += 1

            # ---- fast: controller tracks each waypoint ----
            for k in range(c.exec_h):
                w = waypoints[k] + self.rng.normal(0, self.explore, 2) if self.explore else waypoints[k]
                step = w - env.grip_xy
                dist = np.linalg.norm(step)
                if dist > c.max_step:
                    w = env.grip_xy + step * c.max_step / dist
                label = None
                if rec and corrections:
                    label, oracle = expert_action(env, self.rng)
                    rec.observe(oracle)
                elif rec:
                    rec.observe(sg)
                ctrl.track(env, w, bool(grip[k]))
                if rec:
                    rec.act([w[0], w[1], float(grip[k])], label)
                t += 1
                if env.done or t >= c.max_actions:
                    break
            chunks += 1
            if env.done:
                break

            # ---- events: decide whether the planner must run again ----
            z, prop = self._see(env)
            if self.critic:
                if self._done_prob(z, prop, sg) > c.done_threshold:
                    need_plan, by_done = True, True
                elif prev is not None and self._done_prob(z, prop, prev) < c.regress_threshold:
                    need_plan, prev = True, None
                elif chunks >= c.max_chunks_per_subgoal or vetoes >= c.max_vetoes:
                    need_plan = True
            else:
                need_plan = chunks >= self.timer or vetoes >= c.max_vetoes

        stats.update(success=env.success, failed=env.failed, actions=t)
        if rec:
            rec.observe(oracle_subgoal(env) if corrections else (sg if sg is not None else (DONE, 0)))
            return stats, rec.finish()
        return stats, None


VARIANTS = {
    "planner on a timer": dict(critic=False, rollout_check=False),
    "planner every chunk": dict(critic=False, rollout_check=False, timer=1),
    "+ critic": dict(critic=True, rollout_check=False),
    "+ critic + rollout check": dict(critic=True, rollout_check=True),
    "full, words only": dict(critic=True, rollout_check=True, latent=False),
    "full, latents only": dict(critic=True, rollout_check=True, words=False),
}


def evaluate(cfg, models, n=200, variants=None, seed0=9_000_000):
    """Runs every variant on the same held-out seeds."""
    out = {}
    for name, kw in (variants or VARIANTS).items():
        agent = Agent(cfg, models, **kw)
        runs = [agent.run(ToyKitchen(cfg, seed0 + i))[0] for i in range(n)]
        out[name] = dict(
            success=float(np.mean([r["success"] for r in runs])),
            irreversible_failure=float(np.mean([r["failed"] for r in runs])),
            stopped_early=float(np.mean([r["stopped_early"] and not r["success"] and not r["failed"] for r in runs])),
            timeout=float(np.mean([not r["success"] and not r["failed"] and not r["stopped_early"] for r in runs])),
            planner_calls=float(np.mean([r["planner_calls"] for r in runs])),
            vetoes=float(np.mean([r["vetoes"] for r in runs])),
            actions=float(np.mean([r["actions"] for r in runs])),
            episodes=n,
        )
    return out


def format_table(results):
    cols = ["success", "irreversible_failure", "timeout", "stopped_early", "planner_calls", "vetoes", "actions"]
    head = ["variant", "success", "wrong bin", "timeout", "stopped early", "planner calls", "vetoes", "actions"]
    rows = [head] + [[k] + [f"{v[c]:.2f}" for c in cols] for k, v in results.items()]
    widths = [max(len(r[i]) for r in rows) for i in range(len(head))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(r, widths, strict=True)) for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)
