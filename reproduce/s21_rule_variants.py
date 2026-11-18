"""Supplementary S2.1: rule variants on 3,000 validation scenarios (seeds 50,000 to 52,999), h = 0.2.
Return = the training reward summed over the episode, as in the paper. Writes out/S21_rule_variants.csv."""
import sys, time
sys.path.insert(0, '.')
from v6core import *

torch.set_num_threads(1)
ck = torch.load('models/predictor_env4480df20_v6e447b6bc.pt', map_location='cpu', weights_only=False)
predictor = IntentPredictor(ck['state'], ck['temp'], ck['win'])
H, N, SEED0 = 0.2, 3000, 50_000


class RuleVar(RiskRule):
    """The rule of the paper with margin m (2 m in the paper). With tau_hi, intentions between tau and tau_hi hold the
    speed instead of braking."""
    def __init__(self, tau, m=2.0, tau_hi=None, oracle=False):
        super().__init__(tau, oracle); self.m, self.tau_hi = m, tau_hi

    def __call__(self, env, obs):
        c = env.cfg; gap = env._front_gap()
        if gap < 0 or env.est is None:
            return 0
        x, vx = env.est
        if x < c.xc - c.road_half:
            return 0
        on_road = abs(x - c.xc) <= c.road_half
        t_pass = (max(gap, 0.0) + c.l + c.ped_r) / max(env.sv, 0.5 * env.v_cruise)
        conflict = on_road or (vx < -0.5 and (x - (c.xc + c.road_half)) / (-vx) <= t_pass + 1.0)
        stop = env.sv ** 2 / (2 * c.a_brake) + env.sv * c.dt * c.action_repeat
        if not (conflict and gap - STOP_MARGIN <= stop + self.m):
            return 0
        if on_road or (env.distracted if self.oracle else env.p_used > (self.tau_hi if self.tau_hi else self.tau)):
            return 2
        if self.tau_hi and env.p_used > self.tau:
            return 1
        return 0


def run(policy):
    env = V6Env(H, v2x=True, predictor=predictor); rows = []
    for ep in range(N):
        obs, _ = env.reset(seed=SEED0 + ep); policy.reset(); done = False; ret = 0.0
        while not done:
            obs, r, te, tr, info = env.step(policy(env, obs)); ret += r; done = te or tr
        rows.append(dict(collision=bool(info['collision']), passed=bool(info['passed']), time=info['t'], ret=ret))
    return pd.DataFrame(rows)


if __name__ == '__main__':
    out = 'out/S21_rule_variants.csv'
    done_rows = pd.read_csv(out).to_dict('records') if os.path.exists(out) else []
    have = {(r['variant']) for r in done_rows}
    variants = [(f'tau={t}, m={m}', RuleVar(t, m)) for t in (0.03, 0.05, 0.08) for m in (0, 0.5, 1, 2)]
    variants += [(f'hold 0.05 to {th}', RuleVar(0.05, 2.0, th)) for th in (0.1, 0.2, 0.3)]
    variants += [('oracle rule', RuleVar(0, 2.0, oracle=True))]
    for name, pol in variants:
        if name in have:
            continue
        t0 = time.time(); d = run(pol)
        done_rows.append(dict(variant=name, collisions_pct=100 * d.collision.mean(), time_s=d[d.passed].time.mean(), mean_return=d.ret.mean()))
        pd.DataFrame(done_rows).to_csv(out, index=False)
        print(f'{name:22s} {100*d.collision.mean():.2f}%  {d[d.passed].time.mean():.2f} s  return {d.ret.mean():.2f}  ({time.time()-t0:.0f} s)', flush=True)
