"""Supplementary S2.2, last step: a fine-tuned network that beat its start network on validation is compared with that
start on 3,000 other scenarios (seeds 51,000 to 53,999). Each network is used in the mode of its best validation
(greedy or sampled). Writes out/S22_test.csv. Usage: python s22_test.py [--all]"""
import sys, json, glob, time
sys.path.insert(0, '.')
from v6core import *

torch.set_num_threads(1)
ck = torch.load('models/predictor_env4480df20_v6e447b6bc.pt', map_location='cpu', weights_only=False)
predictor = IntentPredictor(ck['state'], ck['temp'], ck['win'])
H, N, SEED0 = 0.2, 3000, 51_000


def run(model, mode):
    env = V6Env(H, v2x=True, predictor=predictor); pol = policy_for(model, mode); rows = []
    for ep in range(N):
        obs, _ = env.reset(seed=SEED0 + ep); done = False; ret = 0.0
        pol.reset(SEED0 + ep) if isinstance(pol, SampledPolicy) else pol.reset()
        while not done:
            obs, r, te, tr, info = env.step(pol(env, obs)); ret += r; done = te or tr
        rows.append(dict(collision=bool(info['collision']), passed=bool(info['passed']), time=info['t'], ret=ret))
    d = pd.DataFrame(rows)
    return dict(mean_return=d.ret.mean(), collisions_pct=100 * d.collision.mean(), time_s=d[d.passed].time.mean()), d.ret.values


if __name__ == '__main__':
    out, rows, starts = 'out/S22_test.csv', [], {}
    for f in sorted(glob.glob('out/S22_scheme*_run*.json')):
        r = json.load(open(f))
        best = max(r['best_val_sampled'], r['best_val_greedy']); start = r['start_val_greedy'] if r['start_mode'] == 'greedy' else r['start_val_sampled']
        print(f"scheme {r['scheme']}, run {r['run']}: best validation {best:.2f} against {start:.2f} for the start", flush=True)
        if best <= start and '--all' not in sys.argv:
            continue
        t0 = time.time()
        if r['run'] not in starts:
            m0, c0 = load_v6(f"models/PPOV2Xintention_run{r['run']}.pt"); starts[r['run']] = run(m0, c0['extra']['mode'])
        m1, _ = load_v6(f"models_s22/scheme{r['scheme']}_run{r['run']}.pt")
        mode1 = 'sampled' if r['best_val_sampled'] >= r['best_val_greedy'] else 'greedy'
        s1, ret1 = run(m1, mode1); s0, ret0 = starts[r['run']]
        from scipy.stats import wilcoxon
        p = wilcoxon(ret1, ret0).pvalue if np.any(ret1 != ret0) else 1.0
        rows.append(dict(scheme=r['scheme'], run=r['run'], mode=mode1, **{f'tuned_{k}': v for k, v in s1.items()},
                         **{f'start_{k}': v for k, v in s0.items()}, p_return=p))
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  tuned ({mode1}): {s1}\n  start: {s0}\n  paired test of returns: p = {p:.3f}  ({time.time()-t0:.0f} s)", flush=True)
