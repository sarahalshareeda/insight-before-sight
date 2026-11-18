"""Control runs (imitation start, no V2X): Cell C2 of the RL notebook (notebooks/rl_cell_control_run.py), run on CPU.
Same settings as Cell B: TRAIN_H = 0.2, teacher tau = 0.05, ppo_cfg and ft_cfg as in Cell B. Usage: python train_control.py 0 2 4"""
import sys, copy, time
sys.path.insert(0, '.')
import v6core
from v6core import *
torch.set_num_threads(1)
ck = torch.load('models/predictor_env4480df20_v6e447b6bc.pt', map_location='cpu', weights_only=False)
predictor = IntentPredictor(ck['state'], ck['temp'], ck['win'])
TRAIN_H, TEACHER_TAU, vcfg = 0.2, 0.05, V6Config()
ppo_cfg = PPOConfig(reward_scale=0.02, val_episodes=1000, val_every=40, total_frames=1_200_000)
ft_cfg = replace(ppo_cfg, lr=1e-4, ent_start=0.01, ent_end=0.001, total_frames=600_000)
assert ppo_cfg.key() == '8cd5154a', ppo_cfg.key()
CTRL_NAME = "Imitation start, no V2X"
_V6Env = v6core.V6Env
class _NoV2XEnv(_V6Env):
    def __init__(self, hesitation_prob=0.2, v2x=True, **kw):
        super().__init__(hesitation_prob, v2x=False, **kw)
make_c = lambda: _V6Env(TRAIN_H, v2x=False, predictor=predictor, vcfg=vcfg)
os.makedirs('models_control', exist_ok=True)
for run in map(int, sys.argv[1:]):
    path = os.path.join('models_control', f"{slug(CTRL_NAME)}_run{run}.pt")
    if os.path.exists(path):
        print("found", path); continue
    t0 = time.time(); print(f"=== {CTRL_NAME}, run {run} ===", flush=True)
    ipath = os.path.join('models_control', f"imitation_noV2X_run{run}.pt")
    if os.path.exists(ipath):
        imitation = LSTMActorCritic(11, ppo_cfg.hidden, 3).to(device); imitation.load_state_dict(torch.load(ipath, map_location=device))
    else:
        v6core.V6Env = _NoV2XEnv          # imitation_start builds its demonstrations with V6Env
        try:
            imitation = imitation_start(predictor, TEACHER_TAU, ppo_cfg, seed0=300_000 + 10_000 * run, vcfg=vcfg, seed=run)
        finally:
            v6core.V6Env = _V6Env
        torch.save(imitation.state_dict(), ipath)
    print(f"  imitation done {time.time()-t0:.0f} s", flush=True)
    cands = [("imitation", mode, val_return(imitation, make_c, ppo_cfg, mode), copy.deepcopy(imitation)) for mode in ("greedy", "sampled")]
    model = copy.deepcopy(imitation)
    log = train_ppo(make_c, model, ft_cfg, seed=run, log_every=60, val_policy=SampledPolicy)
    cands.append(("PPO", "sampled", log.get("best_val_return", -1e9), model))
    cands.append(("PPO", "greedy", val_return(model, make_c, ppo_cfg, "greedy"), model))
    src, mode, val, best = max(cands, key=lambda c: c[2])
    print("  validation:", [(c[0], c[1], round(c[2], 2)) for c in cands], "->", src, mode, f"({time.time()-t0:.0f} s)", flush=True)
    save_ckpt(path, best, log, variant=CTRL_NAME, run=run, start="imitation", source=src, mode=mode, val=val,
              candidates=[(c[0], c[1], c[2]) for c in cands])
