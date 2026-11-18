"""Tests of the paper on a CPU: T0 to T9 (see the README for what each part covers).

Uses the definition cells of notebooks/rl_agent.ipynb (v6core.py, copied verbatim), the final models
(threshold 0.05) and the trained predictor. Every test uses the paper's 2,000 test scenarios (seeds 10,000+) at
h = 0.2 unless stated. Each part writes its episodes and a summary to OUT and is skipped if its file exists.

Usage: python rerun_tests.py T0 T1 T2 T3 T4 T5
"""
import sys, time
sys.path.insert(0, '.')
from v6core import *

torch.set_num_threads(1)
OUT = 'out'
os.makedirs(OUT, exist_ok=True)
ck = torch.load('models/predictor_env4480df20_v6e447b6bc.pt', map_location='cpu', weights_only=False)
predictor = IntentPredictor(ck['state'], ck['temp'], ck['win'])
TAU_STAR, N, H = 0.05, 2000, 0.2
VC = V6Config()
assert ENV6.key() == '4480df20' and VC.key() == 'e447b6bc'


def load_runs(prefix, n):
    runs = []
    for r in range(n):
        m, c = load_v6(f'models/{prefix}_run{r}.pt')
        runs.append((r, policy_for(m, c['extra']['mode'])))
    return runs


V2X = load_runs('PPOV2Xintention', 5)
NOV2X = load_runs('PPOnoV2X', 5)
FLOOR = ConstPolicy(0)


def run_env(env, policy, episodes=N, seed0=10_000):
    """evaluate6 of the notebook, with any environment object."""
    rows = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        policy.reset(seed0 + ep) if isinstance(policy, SampledPolicy) else policy.reset()
        done = False; nb = na = 0
        while not done:
            a = policy(env, obs); nb += int(a == 2); na += 1
            obs, r, te, tr, info = env.step(a); done = te or tr
        rows.append(dict(ep=seed0 + ep, collision=bool(info["collision"]), passed=bool(info["passed"]),
                         timeout=bool(info["time_limit"]), near_miss=bool(info["near_miss"]), time=info["t"],
                         min_dist=info["min_dist"], brake_share=nb / na, distracted=bool(env.distracted)))
    return pd.DataFrame(rows)


class BurstyEnv(V6Env):
    """Gilbert-Elliott V2X loss (paper, Section II-E): loss eps_g in the good state and eps_b in the bad state.
    Mean burst length burst_s; the bad-state share is set so that the mean loss equals mean_loss."""
    def __init__(self, h, mean_loss, burst_s, eps_g=0.01, eps_b=1.0, **kw):
        super().__init__(h, **kw)
        self.eps_g, self.eps_b = eps_g, eps_b
        self.pi_b = (mean_loss - eps_g) / (eps_b - eps_g)
        self.p_bg = self.cfg.dt / burst_s
        self.p_gb = self.pi_b * self.p_bg / (1.0 - self.pi_b)

    def _reset_inference(self):
        super()._reset_inference()
        self.bad = bool(self.noise_rng.random() < self.pi_b)      # start in the stationary distribution

    def _v2x_send(self, *args):
        if not self.enable_v2x:
            return
        r = self.noise_rng
        self.bad = (r.random() >= self.p_bg) if self.bad else (r.random() < self.p_gb)
        if r.random() < (self.eps_b if self.bad else self.eps_g):
            return
        d, vx = self.rsu_track[-1]
        delay = int(r.integers(0, 2)) + self.v.extra_delay
        self.rsu_q.append((self.t + delay, (self.t, d + self._curb(), vx, self.p_rsu)))


ALL_EPS, ALL_SUM = {}, {}


CACHE = os.path.join(OUT, 'cache')
os.makedirs(CACHE, exist_ok=True)


def record(part, condition, policy, run, df):
    # df may be a function: its result is cached per evaluation, so a stopped run resumes where it stopped
    if callable(df):
        key = ''.join(ch if ch.isalnum() else '_' for ch in f'{part}__{condition}__{policy}__{run}')
        path = os.path.join(CACHE, key + '.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            df = df(); df.to_csv(path + '.tmp', index=False); os.replace(path + '.tmp', path)
    d = df.copy(); d['condition'], d['policy'], d['run'] = condition, policy, run
    ALL_EPS.setdefault(part, []).append(d)
    ALL_SUM.setdefault(part, []).append(dict(condition=condition, policy=policy, run=run, **summarize(df)))
    s = ALL_SUM[part][-1]
    print(f'  {part} | {condition:28s} | {policy:34s} | run {run} | {s["collision_pct"]:.2f}% '
          f'[{s["ci_low"]:.2f}, {s["ci_high"]:.2f}] | {s["time_to_pass_s"]:.2f} s | brake {s["brake_share_pct"]:.1f}%', flush=True)


def save(part):
    pd.concat(ALL_EPS[part], ignore_index=True).to_csv(f'{OUT}/{part}_episodes.csv', index=False)
    pd.DataFrame(ALL_SUM[part]).to_csv(f'{OUT}/{part}_summary.csv', index=False)


def done(part):
    return os.path.exists(f'{OUT}/{part}_summary.csv')


def T0():
    """Nominal channel, h = 0.2: must equal the Colab results (episodes_v6.csv) scenario by scenario."""
    p = 'T0_nominal_check'
    for r, pol in V2X: record(p, 'nominal', 'PPO + V2X intention', r, lambda: run_env(V6Env(H, v2x=True, predictor=predictor), pol))
    for r, pol in NOV2X: record(p, 'nominal', 'PPO, no V2X', r, lambda: run_env(V6Env(H, v2x=False, predictor=None), pol))
    record(p, 'nominal', 'Car alone (safety floor)', None, lambda: run_env(V6Env(H, v2x=False, predictor=None), FLOOR))
    record(p, 'nominal', f'Rule: V2X intention > {TAU_STAR}', None, lambda: run_env(V6Env(H, v2x=True, predictor=predictor), RiskRule(TAU_STAR)))
    record(p, 'nominal', 'Rule: brake for every approaching pedestrian', None, lambda: run_env(V6Env(H, v2x=True, predictor=predictor), RiskRule(0)))
    save(p)


FAULTS = [('10% loss', dict(v2x_loss=0.10)), ('30% loss', dict(v2x_loss=0.30)), ('RSU down', dict(v2x_loss=1.0)),
          ('+300 ms delay', dict(extra_delay=3)), ('+500 ms delay', dict(extra_delay=5))]


def T1():
    """Channel faults for all five RL + V2X runs and the calibrated rule."""
    p = 'T1_faults'
    for lab, kw in FAULTS:
        vc = replace(VC, **kw)
        for r, pol in V2X: record(p, lab, 'PPO + V2X intention', r, lambda: run_env(V6Env(H, v2x=True, predictor=predictor, vcfg=vc), pol))
        record(p, lab, f'Rule: V2X intention > {TAU_STAR}', None, lambda: run_env(V6Env(H, v2x=True, predictor=predictor, vcfg=vc), RiskRule(TAU_STAR)))
    save(p)


BURSTS = [(0.10, 0.5), (0.30, 0.5), (0.10, 2.0), (0.30, 2.0)]


def T2():
    """Bursty loss (Gilbert-Elliott), mean loss 10% or 30%, mean bursts of 0.5 or 2 sec."""
    p = 'T2_bursty'
    for L, B in BURSTS:
        lab = f'bursty {int(L*100)}% loss, {B} s bursts'
        for r, pol in V2X: record(p, lab, 'PPO + V2X intention', r, lambda: run_env(BurstyEnv(H, L, B, v2x=True, predictor=predictor, vcfg=VC), pol))
        record(p, lab, f'Rule: V2X intention > {TAU_STAR}', None, lambda: run_env(BurstyEnv(H, L, B, v2x=True, predictor=predictor, vcfg=VC), RiskRule(TAU_STAR)))
    save(p)


TAUS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]


def T3():
    """Weaker pedestrian cue: slowdown factor beta = 0.85 and 1 (none) instead of 0.74."""
    p = 'T3_cue'
    rep_rows = []
    for beta in (0.85, 1.0):
        vc = replace(VC, appraise_factor=beta)
        lab = f'beta = {beta}, trained predictor'
        for r, pol in V2X: record(p, lab, 'PPO + V2X intention', r, lambda: run_env(V6Env(H, v2x=True, predictor=predictor, vcfg=vc), pol))
        record(p, lab, f'Rule: V2X intention > {TAU_STAR}', None, lambda: run_env(V6Env(H, v2x=True, predictor=predictor, vcfg=vc), RiskRule(TAU_STAR)))
        record(p, lab, 'Rule: brake for every approaching pedestrian', None, lambda: run_env(V6Env(H, v2x=True, predictor=predictor, vcfg=vc), RiskRule(0)))
        # retrained predictor (Cell V2 with the same data size and seeds) and recalibrated threshold (Cell C)
        pc = os.path.join(CACHE, f'predictor_beta{beta}.pt')
        if os.path.exists(pc):
            z = torch.load(pc, weights_only=False); state, temp, rep = z['state'], z['temp'], z['rep']
        else:
            X, y, d, e = predictor_data(500, 200_000, vcfg=vc)
            net, temp, rep = train_predictor(X, y, d, e, vc.win, verbose=True)
            state = {k: v.cpu() for k, v in net.state_dict().items()}
            torch.save(dict(state=state, temp=temp, rep=rep), pc)
        pred_b = IntentPredictor(state, temp, vc.win)
        best, cal = None, []
        for tau in TAUS:
            cpath = os.path.join(CACHE, f'calib_beta{beta}_tau{tau}.csv')
            if os.path.exists(cpath):
                df = pd.read_csv(cpath)
            else:
                df = run_env(V6Env(H, v2x=True, predictor=pred_b, vcfg=vc), RiskRule(tau), N, 20_000); df.to_csv(cpath, index=False)
            k = int(df.collision.sum()); from scipy.stats import beta as B_
            ucb = 100 * B_.ppf(0.95, k + 1, N - k)
            cal.append(dict(beta=beta, tau=tau, collisions=k, ucb_pct=ucb, time_to_pass_s=df[df.passed].time.mean()))
            if ucb <= 1.0:
                best = tau
        rep_rows.append(dict(beta=beta, tau_star=best, **rep))
        pd.DataFrame(cal).to_csv(f'{OUT}/T3_calibration_beta{beta}.csv', index=False)
        lab = f'beta = {beta}, retrained predictor'
        record(p, lab, f'Rule: V2X intention > {best} (recalibrated)', None,
               run_env(V6Env(H, v2x=True, predictor=pred_b, vcfg=vc), RiskRule(best)))
    pd.DataFrame(rep_rows).to_csv(f'{OUT}/T3_predictor_reports.csv', index=False)
    save(p)


def T4():
    """Second set of 2,000 scenarios (seeds 40,000 to 41,999): nominal channel and RSU failure."""
    p = 'T4_second_set'
    for lab, kw in (('nominal', {}), ('RSU down', dict(v2x_loss=1.0))):
        vc = replace(VC, **kw)
        record(p, lab, 'Car alone (safety floor)', None, lambda: run_env(V6Env(H, v2x=False, predictor=None), FLOOR, N, 40_000))
        record(p, lab, f'Rule: V2X intention > {TAU_STAR}', None, lambda: run_env(V6Env(H, v2x=True, predictor=predictor, vcfg=vc), RiskRule(TAU_STAR), N, 40_000))
        for r, pol in V2X: record(p, lab, 'PPO + V2X intention', r, lambda: run_env(V6Env(H, v2x=True, predictor=predictor, vcfg=vc), pol, N, 40_000))
    save(p)


def trace(env, act, seed):
    """Every 0.1 s: time, car and pedestrian positions, speed, visibility, intentions, and the decision."""
    obs, _ = env.reset(seed=seed); rows, done = [], False
    while not done:
        a = act(env, obs)
        for _ in range(env.cfg.action_repeat):
            c = env.cfg
            rows.append(dict(t=round(env.t * c.dt, 1), yv=env.yv, xv=env.xv, sv=env.sv, xp=env.xp, yp=env.yp,
                             gap=env._front_gap(), visible=bool(env._visible()), p_rsu=env.p_rsu, p_used=env.p_used,
                             v2x_fresh=bool(env.rsu_fresh), action=a, distracted=bool(env.distracted)))
            fa = safety_floor(env, env.ACTIONS[int(a)])
            rows[-1]['floor_brake'] = bool(fa == 1 and env.ACTIONS[int(a)] != 1)
            r, te, tr, info = env._substep(fa)
            if te or tr:
                break
        obs = env._obs(); done = te or tr
    return pd.DataFrame(rows), info


def T5(seed=10_798):
    """Fig. 10: test scenario 10,798 driven by RL run 4 and by LLM variant C (replayed from its logged decisions)."""
    p = 'T5_replay'
    os.makedirs(OUT, exist_ok=True)
    m, c = load_v6('models/PPOV2Xintention_run4.pt'); pol = policy_for(m, c['extra']['mode']); pol.reset(seed) if isinstance(pol, SampledPolicy) else pol.reset()
    tr_rl, info_rl = trace(V6Env(H, v2x=True, predictor=predictor), lambda e, o: pol(e, o), seed)
    log = pd.read_csv('data/decisions_ablation_540ad8_h0.2.csv')
    log = log[log.ep == seed]
    acts = {round(t, 1): int(a) for t, a in zip(log.t, log.action)}
    tr_llm, info_llm = trace(V6Env(H, v2x=True, predictor=predictor), lambda e, o: acts.get(round(e.t * e.cfg.dt, 1), 0), seed)
    tr_rl['agent'], tr_llm['agent'] = 'RL + V2X, run 4', 'LLM + V2X, variant C'
    pd.concat([tr_rl, tr_llm]).to_csv(f'{OUT}/{p}_trace.csv', index=False)
    pd.DataFrame([dict(agent='RL + V2X, run 4', **{k: info_rl[k] for k in ('collision', 'passed', 't')}),
                  dict(agent='LLM + V2X, variant C', **{k: info_llm[k] for k in ('collision', 'passed', 't')}),
                  dict(agent='LLM log: decisions found', collision=None, passed=None, t=len(acts))]).to_csv(f'{OUT}/{p}_summary.csv', index=False)
    print(open(f'{OUT}/{p}_summary.csv').read())


S_C = 6.9      # defensive speed (paper, Table I): D(s_c) = d_o - d_s


class DefensivePolicy:
    """Onboard sensors only (no V2X). Before the crosswalk, slow to s_c and hold the speed at or below s_c; past the
    crosswalk line, resume the cruise speed. The safety floor stays on, as for every policy."""
    def reset(self):
        pass

    def __call__(self, env, obs):
        if env._front_gap() <= 0:
            return 0
        if env.sv > S_C:
            return 2
        if env.sv + env.cfg.a_go * env.cfg.dt * env.cfg.action_repeat <= S_C:
            return 0
        return 1


def T6():
    """Defensive baseline of Table III (h = 0.2) and the oracle rule."""
    p = 'T6_defensive'
    record(p, 'nominal', 'Floor + defensive speed', None, lambda: run_env(V6Env(H, v2x=False, predictor=None), DefensivePolicy()))
    record(p, 'nominal', 'Perfect intention (oracle rule)', None, lambda: run_env(V6Env(H, v2x=True, predictor=predictor), RiskRule(0, oracle=True)))
    save(p)


def _conflict_est(env):
    """As in the LLM notebook: the car's estimate shows the pedestrian on the road, or reaching it before the car passes."""
    c, gap = env.cfg, env._front_gap()
    x, vx = env.est
    t_pass = (max(gap, 0.0) + c.l + c.ped_r) / max(env.sv, 0.5 * env.v_cruise)
    return abs(x - c.xc) <= c.road_half or (vx < -0.5 and (x - (c.xc + c.road_half)) / (-vx) <= t_pass + 1.0)


def T7():
    """Decision logs of the five RL + V2X runs and the calibrated rule at h = 0.2 (for Fig. 6), with the same fields
    as the LLM decision logs."""
    rows, ref = [], RiskRule(TAU_STAR)
    for lab, run, pol in [('RL + V2X', r, p_) for r, p_ in V2X] + [('Rule, tau* = 0.05', None, RiskRule(TAU_STAR))]:
        env = V6Env(H, v2x=True, predictor=predictor)
        for ep in range(N):
            obs, _ = env.reset(seed=10_000 + ep)
            pol.reset(10_000 + ep) if isinstance(pol, SampledPolicy) else pol.reset()
            done = False
            while not done:
                a = pol(env, obs)
                if env.est is not None and env.est[0] >= env.cfg.xc - env.cfg.road_half:
                    x, vx = env.est
                    rows.append(dict(agent=lab, run=run, ep=10_000 + ep, t=round(env.t * env.cfg.dt, 1), gap=env._front_gap(),
                                     speed=env.sv, ped_from_road=x - (env.cfg.xc + env.cfg.road_half), ped_vx=vx,
                                     p_used=env.p_used, visible=bool(env._visible()), conflict=bool(_conflict_est(env)),
                                     distracted=bool(env.distracted), action=a, rule_action=ref(env, None)))
                obs, r, te, tr, info = env.step(a); done = te or tr
        print('  T7', lab, run, len(rows), flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/T7_decisions_rl_rule.csv', index=False)
    pd.DataFrame([dict(rows=len(rows))]).to_csv(f'{OUT}/T7_decisions_summary.csv', index=False)


def T8():
    """Held-out test windows of the trained predictor (Fig. 5): the same data and split as get_predictor/train_predictor."""
    X, y, dist, ep_id = predictor_data(500, 200_000, vcfg=VC)
    u = np.unique(ep_id); np.random.default_rng(0).shuffle(u)
    tr_e, ca_e = set(u[:int(.7 * len(u))]), set(u[int(.7 * len(u)):int(.85 * len(u))])
    m_te = np.array([(e not in tr_e) and (e not in ca_e) for e in ep_id])
    W = predictor.W
    def prob(x):
        h = np.maximum(W[0] @ x + W[1], 0.0); h = np.maximum(W[2] @ h + W[3], 0.0)
        return 1.0 / (1.0 + math.exp(-float((W[4] @ h + W[5]).item()) / predictor.temp))
    p = np.array([prob(x) for x in X[m_te]])
    pd.DataFrame(dict(ep=ep_id[m_te], dist=dist[m_te], y=y[m_te], p=p)).to_csv(f'{OUT}/T8_predictor_test_windows.csv', index=False)
    from sklearn.metrics import roc_auc_score
    print(f'  T8: {m_te.sum()} test windows from {len(set(ep_id[m_te]))} episodes, AUC {roc_auc_score(y[m_te], p):.3f}')
    pd.DataFrame([dict(windows=int(m_te.sum()), episodes=len(set(ep_id[m_te])), auc=roc_auc_score(y[m_te], p))]).to_csv(f'{OUT}/T8_summary.csv', index=False)


def T9():
    """Control runs (imitation start, no V2X), as evaluated in Cell C2: all runs at h = 0.2, the run chosen on
    validation at every h, and the calibrated rule acting on the onboard estimate only."""
    p = 'T9_control'
    runs = []
    for r in range(5):
        m, c = load_v6(f'models_control/ImitationstartnoV2X_run{r}.pt')
        runs.append((c['extra']['val'], r, policy_for(m, c['extra']['mode']), c['extra']['source'], c['extra']['mode']))
    c_run = max(runs, key=lambda x: x[0])[1]
    pd.DataFrame([dict(run=r, val=v, source=s, mode=md, chosen=(r == c_run)) for v, r, _, s, md in runs]).to_csv(f'{OUT}/T9_control_runs.csv', index=False)
    eps = []
    for h in (0.0, 0.2, 0.4, 0.6, 0.8):
        for lab, pol, run in [(f'Rule: own intention > {TAU_STAR}, no V2X', RiskRule(TAU_STAR), None)] + \
                             [('Imitation start, no V2X', pp, r) for _, r, pp, _, _ in runs if h == 0.2 or r == c_run]:
            record(p, f'h = {h}', lab, run, lambda: run_env(V6Env(h, v2x=False, predictor=predictor), pol))
            ALL_EPS[p][-1]['h'] = h
    save(p)


if __name__ == '__main__':
    for part in sys.argv[1:]:
        names = {'T0': 'T0_nominal_check', 'T1': 'T1_faults', 'T2': 'T2_bursty', 'T3': 'T3_cue', 'T4': 'T4_second_set', 'T5': 'T5_replay',
                 'T6': 'T6_defensive', 'T7': 'T7_decisions', 'T8': 'T8', 'T9': 'T9_control'}
        if done(names[part]):
            print('skip', part); continue
        t0 = time.time(); print(f'=== {part} ===', flush=True)
        globals()[part]()
        print(f'=== {part} done in {time.time() - t0:.0f} s ===', flush=True)
