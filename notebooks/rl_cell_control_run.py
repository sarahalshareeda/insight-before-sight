# ===== CELL C2: control for the imitation start (run after Cell D; resumable)
# Question: is the gain of "PPO + V2X intention" due to V2X, or due to imitating the rule?
# Same pipeline as Cell B, without V2X: the car runs the predictor on its own track only (p = p_C), imitates the
# calibrated rule acting on p_C, PPO may fine-tune it, and each run keeps whichever is best on the validation
# scenarios. The calibrated rule without V2X is also tested. Results go to new files; nothing earlier is changed.
import copy
CTRL_NAME = "Imitation start, no V2X"
CTRL_RUNS = 5
CTRL_FINETUNE = True          # False skips PPO fine-tuning (about 2 h faster); True follows Cell B exactly
RULE_NOV2X = f"Rule: own intention > {TEACHER_TAU}, no V2X"

_V6Env = V6Env
class _NoV2XEnv(_V6Env):      # the same environment with V2X switched off, used only while imitating
    def __init__(self, hesitation_prob=0.2, v2x=True, **kw):
        super().__init__(hesitation_prob, v2x=False, **kw)

make_c = lambda: _V6Env(TRAIN_H, v2x=False, predictor=predictor, vcfg=vcfg)
for run in range(CTRL_RUNS):
    path = os.path.join(run_dir, f"{slug(CTRL_NAME)}_run{run}.pt")
    if os.path.exists(path):
        print("found", path); continue
    print(f"\n=== {CTRL_NAME}, run {run} ===")
    ipath = os.path.join(run_dir, f"imitation_noV2X_run{run}.pt")
    if os.path.exists(ipath):
        imitation = LSTMActorCritic(11, ppo_cfg.hidden, 3).to(device); imitation.load_state_dict(torch.load(ipath, map_location=device))
    else:
        globals()["V6Env"] = _NoV2XEnv          # imitation_start builds its demonstration episodes with V6Env
        try:
            imitation = imitation_start(predictor, TEACHER_TAU, ppo_cfg, seed0=300_000 + 10_000 * run, vcfg=vcfg, seed=run)
        finally:
            globals()["V6Env"] = _V6Env
        torch.save(imitation.state_dict(), ipath)
    cands = [("imitation", mode, val_return(imitation, make_c, ppo_cfg, mode), copy.deepcopy(imitation)) for mode in ("greedy", "sampled")]
    log = {}
    if CTRL_FINETUNE:
        model = copy.deepcopy(imitation)
        log = train_ppo(make_c, model, ft_cfg, seed=run, log_every=60, val_policy=SampledPolicy)
        cands.append(("PPO", "sampled", log.get("best_val_return", -1e9), model))
        cands.append(("PPO", "greedy", val_return(model, make_c, ppo_cfg, "greedy"), model))
    src, mode, val, best = max(cands, key=lambda c: c[2])           # validation scenarios only
    print("  validation:", [(c[0], c[1], round(c[2], 2)) for c in cands], "->", src, mode)
    save_ckpt(path, best, log, variant=CTRL_NAME, run=run, start="imitation", source=src, mode=mode, val=val,
              candidates=[(c[0], c[1], c[2]) for c in cands])

# evaluation on the same test scenarios as Cell C: all runs at h = 0.2, the chosen run at every h
ctrl = []
for run in range(CTRL_RUNS):
    m, ck = load_v6(os.path.join(run_dir, f"{slug(CTRL_NAME)}_run{run}.pt"))
    ctrl.append((ck["extra"]["val"], run, policy_for(m, ck["extra"]["mode"]), ck["extra"]["source"], ck["extra"]["mode"]))
c_run = max(ctrl, key=lambda r: r[0])[1]
print(f"{CTRL_NAME}: " + ", ".join(f"run {r[1]}: val {r[0]:.2f} ({r[3]}, {r[4]})" for r in ctrl) + f" -> run {c_run}")
rows_c, eps_c = [], []
for h in EVAL_H:
    for lab, pol, run in [(RULE_NOV2X, RiskRule(TEACHER_TAU), None)] + [(CTRL_NAME, p, r) for _, r, p, _, _ in ctrl if h == 0.2 or r == c_run]:
        df = evaluate6(pol, h, False, predictor, TEST_EPISODES)
        rows_c.append(dict(h=h, policy=lab, group="control", run=run, **summarize(df)))
        d = df.copy(); d["h"], d["policy"], d["run"] = h, lab, run; eps_c.append(d)
    print(f"h = {h} done")
summary_c = pd.DataFrame(rows_c); episodes_c = pd.concat(eps_c, ignore_index=True)
summary_c.to_csv(os.path.join(run_dir, "summary_v6_control.csv"), index=False)
episodes_c.to_csv(os.path.join(run_dir, "episodes_v6_control.csv"), index=False)

for h in EVAL_H:
    s = summary_c[summary_c.h == h]
    print(f"\n=== h = {h} ===")
    print(s[(s.run.isna()) | (s.run == c_run)][["policy", "run"] + cols].round(2).to_string(index=False))
    if h == 0.2:
        r = s[s.policy == CTRL_NAME]
        print(f"  {CTRL_NAME}, all {len(r)} runs: collisions {r.collision_pct.mean():.2f} +/- {r.collision_pct.std():.2f}%, "
              f"time {r.time_to_pass_s.mean():.2f} +/- {r.time_to_pass_s.std():.2f} s")
    ctrl_ep = episodes_c[(episodes_c.policy == CTRL_NAME) & (episodes_c.h == h) & (episodes_c.run == c_run)].sort_values("ep")
    rule_ep = episodes_c[(episodes_c.policy == RULE_NOV2X) & (episodes_c.h == h)].sort_values("ep")
    pairs = [("PPO + V2X intention", ep_of("PPO + V2X intention", h, chosen["PPO + V2X intention"]), CTRL_NAME, ctrl_ep)]
    if cal_label:
        pairs.append((cal_label, ep_of(cal_label, h), RULE_NOV2X, rule_ep))
    for la, A_, lb, B_ in pairs:
        n10, n01, p = mcnemar_exact(A_.collision.values, B_.collision.values)
        print(f"  {la} vs {lb}: collisions only in the first {n10}, only in the second {n01}, McNemar p = {p:.4f}")
