"""Supplementary S2.2: further fine-tuning of the RL + V2X agent with a privileged critic.

Start: the selected networks of runs 2 and 4. 600,000 decisions, learning rate 1e-4 (linear decay), entropy 0.01 to
0.001, validation every 40 updates on the 1,000 validation scenarios, as in Cell B. A separate critic receives the 11
inputs of the agent plus the true pedestrian state (distracted flag, position, speed, behavior state, timer, true
conflict, visibility) and is trained alone for 30 updates before the actor. The actor keeps its inputs. The loss adds
beta * KL(pi_0 || pi) toward the start network.
  Scheme 1: beta = 0.1.
  Scheme 2: beta = 0.3; 45% distracted pedestrians in training (intention prior kept at 0.15), each step weighted by
            the likelihood ratio of its episode.
Usage: python s22_privileged_critic.py 1 2   (scheme 1, run 2)"""
import sys, copy, time, json
sys.path.insert(0, '.')
from v6core import *

torch.set_num_threads(1)
ck = torch.load('models/predictor_env4480df20_v6e447b6bc.pt', map_location='cpu', weights_only=False)
predictor = IntentPredictor(ck['state'], ck['temp'], ck['win'])
H = 0.2
base = PPOConfig(reward_scale=0.02, val_episodes=1000, val_every=40, total_frames=1_200_000)
pc = replace(base, lr=1e-4, ent_start=0.01, ent_end=0.001, total_frames=600_000)
STATES = ['approach', 'cross', 'wait', 'back', 'hes_stop', 'hes_step', 'done']
P_TRAIN, P_PRIOR = 0.45, 0.15


def priv(env):
    """True pedestrian state, scaled like the agent inputs (7 values)."""
    c = env.cfg
    st = STATES.index(env.ped_state) / (len(STATES) - 1) if env.ped_state in STATES else 1.0
    return np.array([float(env.distracted), (env.xp - env._curb()) / 10.0, env.sp / 1.5, st,
                     getattr(env, 'ped_timer', 0.0) / 2.0, float(env._conflict(perceived=False)), float(env._visible())], np.float32)


def make_env(scheme):
    if scheme == 2:
        e = V6Env(H, v2x=True, predictor=predictor, cfg=replace(ENV6, p_distracted=P_TRAIN))
        e.prior = P_PRIOR
        return e
    return V6Env(H, v2x=True, predictor=predictor)


def lr_weight(env, scheme):
    if scheme != 2:
        return 1.0
    return P_PRIOR / P_TRAIN if env.distracted else (1 - P_PRIOR) / (1 - P_TRAIN)


def validate(model, mode, episodes=1000, seed0=50_000):
    env = V6Env(H, v2x=True, predictor=predictor); pol = policy_for(model, mode); total = 0.0
    for ep in range(episodes):
        o, _ = env.reset(seed=seed0 + ep); done = False
        pol.reset(seed0 + ep) if isinstance(pol, SampledPolicy) else pol.reset()
        while not done:
            o, r, te, tr, _ = env.step(pol(env, o)); total += r; done = te or tr
    return total / episodes


def train(scheme, run, seed=0):
    beta = 0.1 if scheme == 1 else 0.3
    m0, c0 = load_v6(f'models/PPOV2Xintention_run{run}.pt'); mode0 = c0['extra']['mode']
    actor = copy.deepcopy(m0); actor.train(); pi0 = copy.deepcopy(m0).eval()
    critic = LSTMActorCritic(11 + 7, pc.hidden, 3).to(device)
    opt_a = torch.optim.Adam(actor.parameters(), lr=pc.lr); opt_c = torch.optim.Adam(critic.parameters(), lr=pc.lr)
    torch.manual_seed(seed); np.random.seed(seed)
    N, T, L = pc.num_envs, pc.rollout, pc.seq_len
    envs = [make_env(scheme) for _ in range(N)]
    obs = np.stack([e.reset(seed=seed * 1000 + i)[0] for i, e in enumerate(envs)])
    pv = np.stack([priv(e) for e in envs]); wts = np.array([lr_weight(e, scheme) for e in envs], np.float32)
    ha, hc, hp = actor.init_hidden(N), critic.init_hidden(N), pi0.init_hidden(N)
    start = torch.ones(N); n_updates = pc.total_frames // (N * T); Hd = pc.hidden
    start_val = {md: validate(m0, md) for md in ('greedy', 'sampled')}
    best_val, best_state, best_u, vals = -1e9, None, -1, []
    t0 = time.time()
    print(f'scheme {scheme}, run {run}: start network ({mode0}) validation {start_val}', flush=True)
    for u in range(n_updates):
        frac = max(1.0 - u / n_updates, 0.05)
        for g in opt_a.param_groups + opt_c.param_groups: g['lr'] = pc.lr * frac
        B = dict(o=torch.zeros(T, N, 11), p=torch.zeros(T, N, 7), a=torch.zeros(T, N, dtype=torch.long), lp=torch.zeros(T, N),
                 v=torch.zeros(T + 1, N), r=torch.zeros(T, N), d=torch.zeros(T, N), s=torch.zeros(T, N), w=torch.zeros(T, N),
                 ha=torch.zeros(T, N, Hd), ca=torch.zeros(T, N, Hd), hc=torch.zeros(T, N, Hd), cc=torch.zeros(T, N, Hd),
                 h0=torch.zeros(T, N, Hd), c0=torch.zeros(T, N, Hd))
        with torch.no_grad():
            for t in range(T):
                x = torch.as_tensor(obs); xp = torch.as_tensor(np.concatenate([obs, pv], 1), dtype=torch.float32)
                B['o'][t], B['p'][t], B['s'][t], B['w'][t] = x, torch.as_tensor(pv), start, torch.as_tensor(wts)
                B['ha'][t], B['ca'][t] = ha[0][0], ha[1][0]; B['hc'][t], B['cc'][t] = hc[0][0], hc[1][0]; B['h0'][t], B['c0'][t] = hp[0][0], hp[1][0]
                lg, _, ha = actor(x.view(N, 1, 11), ha, start.view(N, 1)); _, v, hc = critic(xp.view(N, 1, 18), hc, start.view(N, 1))
                _, _, hp = pi0(x.view(N, 1, 11), hp, start.view(N, 1))
                dist = Categorical(logits=lg[:, 0]); a = dist.sample()
                B['a'][t], B['lp'][t], B['v'][t] = a, dist.log_prob(a), v[:, 0]
                for i, e in enumerate(envs):
                    o, r, te, tr, info = e.step(int(a[i])); B['r'][t, i] = r * pc.reward_scale; dn = te or tr; B['d'][t, i] = float(dn)
                    if dn:
                        o, _ = e.reset(); wts[i] = lr_weight(e, scheme)
                    start[i] = float(dn); obs[i] = o; pv[i] = priv(e)
            xp = torch.as_tensor(np.concatenate([obs, pv], 1), dtype=torch.float32)
            _, vl, _ = critic(xp.view(N, 1, 18), hc, start.view(N, 1)); B['v'][T] = vl[:, 0]
        adv = torch.zeros(T, N); last = torch.zeros(N)
        for t in reversed(range(T)):
            nt = 1.0 - B['d'][t]; delta = B['r'][t] + pc.gamma * B['v'][t + 1] * nt - B['v'][t]
            last = delta + pc.gamma * pc.lam * nt * last; adv[t] = last
        ret = adv + B['v'][:T]; S = T // L
        sq = lambda z: z[:S * L].transpose(0, 1).reshape(N * S, L, *z.shape[2:])
        h0s = lambda z: z[:S * L:L].transpose(0, 1).reshape(N * S, Hd)
        so, sp_, sa, slp, sr, sad, ss, sw = map(sq, (B['o'], B['p'], B['a'], B['lp'], ret, adv, B['s'], B['w']))
        Ha, Ca, Hc, Cc, H0, C0 = map(h0s, (B['ha'], B['ca'], B['hc'], B['cc'], B['h0'], B['c0']))
        ent = pc.ent_start + (pc.ent_end - pc.ent_start) * u / max(n_updates - 1, 1)
        critic_only = u < 30
        for _ in range(pc.epochs):
            perm = torch.randperm(N * S)
            for k in range(0, N * S, pc.batch_seqs):
                idx = perm[k:k + pc.batch_seqs]; w = sw[idx]
                _, v, _ = critic(torch.cat([so[idx], sp_[idx]], -1), (Hc[idx].unsqueeze(0), Cc[idx].unsqueeze(0)), ss[idx])
                vf = (w * (sr[idx] - v).pow(2)).sum() / w.sum()
                opt_c.zero_grad(); (pc.vf_coef * vf).backward(); nn.utils.clip_grad_norm_(critic.parameters(), pc.max_grad); opt_c.step()
                if critic_only:
                    continue
                lg, _, _ = actor(so[idx], (Ha[idx].unsqueeze(0), Ca[idx].unsqueeze(0)), ss[idx])
                with torch.no_grad():
                    lg0, _, _ = pi0(so[idx], (H0[idx].unsqueeze(0), C0[idx].unsqueeze(0)), ss[idx])
                dist = Categorical(logits=lg); lp = dist.log_prob(sa[idx])
                am = sad[idx]; am = (am - am.mean()) / (am.std() + 1e-8)
                ratio = torch.exp(lp - slp[idx])
                pg = -(w * torch.min(ratio * am, torch.clamp(ratio, 1 - pc.clip, 1 + pc.clip) * am)).sum() / w.sum()
                p0 = torch.softmax(lg0, -1); kl = (p0 * (torch.log_softmax(lg0, -1) - torch.log_softmax(lg, -1))).sum(-1).mean()
                loss = pg + beta * kl - ent * dist.entropy().mean()
                opt_a.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(actor.parameters(), pc.max_grad); opt_a.step()
        if (u + 1) % pc.val_every == 0 or u == n_updates - 1:
            vr = validate(actor, 'sampled'); vals.append((u + 1, vr))
            print(f'  update {u+1}/{n_updates}: validation {vr:.2f}  ({time.time()-t0:.0f} s)', flush=True)
            if vr > best_val:
                best_val, best_u, best_state = vr, u + 1, {k: v.detach().clone() for k, v in actor.state_dict().items()}
    actor.load_state_dict(best_state)
    gv = validate(actor, 'greedy')
    res = dict(scheme=scheme, run=run, beta=beta, start_mode=mode0, start_val_greedy=start_val['greedy'], start_val_sampled=start_val['sampled'],
               start_val=max(start_val.values()), best_val_sampled=best_val, best_val_greedy=gv, best_update=best_u, val_curve=vals,
               minutes=(time.time() - t0) / 60)
    torch.save(dict(model_state=actor.state_dict(), hidden=pc.hidden, num_actions=3, extra=res), f'models_s22/scheme{scheme}_run{run}.pt')
    json.dump(res, open(f'out/S22_scheme{scheme}_run{run}.json', 'w'), indent=1)
    print('RESULT', json.dumps({k: v for k, v in res.items() if k != 'val_curve'}), flush=True)


if __name__ == '__main__':
    os.makedirs('models_s22', exist_ok=True)
    scheme, run = int(sys.argv[1]), int(sys.argv[2])
    if not os.path.exists(f'out/S22_scheme{scheme}_run{run}.json'):
        train(scheme, run)
