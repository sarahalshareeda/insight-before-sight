"""Figures of the paper, built from the raw outputs.

Inputs: data/ (results from Colab and Drive) and out/ (rerun_tests.py). Outputs: figs/*.pdf at the paper's sizes.
Usage: python make_figures.py fig4 fig5 fig6 fig7 fig8 fig10 figS1 figS3     (no argument: all)
"""
import sys, os, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
from scipy import stats

plt.rcParams.update({'font.family': 'Liberation Sans', 'mathtext.fontset': 'stix', 'font.size': 7,
                     'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42})
RED, PINK, BLUE, ORANGE, DGRAY, LGRAY = '#d62728', '#ed9e9e', '#1f5fbf', '#f28e00', '#555555', '#a6a6a6'
COL_W = 3.5      # inches, one IEEE column
D, O, F = 'data', 'out', 'figs'
os.makedirs(F, exist_ok=True)
RULE, TRACK, ORACLE, FLOOR = 'Rule: V2X intention > 0.05', 'Rule: brake for every approaching pedestrian', 'Perfect intention (upper bound)', 'Car alone (safety floor)'


def cp(k, n):
    lo = stats.beta.ppf(0.025, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) if k < n else 1.0
    return 100 * lo, 100 * hi


def tci(x):
    x = np.asarray(x, float); m = x.mean()
    if len(x) < 2:
        return m, m
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x))
    return m - h, m + h


def rate(df):
    return 100 * df.collision.mean()


def save(fig, name):
    fig.savefig(f'{F}/{name}.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print('saved', f'{F}/{name}.pdf')


# ---------------------------------------------------------------- data access
def ep6():
    return pd.read_csv(f'{D}/episodes_v6.csv')


def llm_c_nominal():
    return pd.read_csv(f'{D}/episodes_ablation_540ad8_h0.2.csv').drop_duplicates('ep')


def llm_c_sweep():
    return pd.read_csv(f'{D}/episodes_sweep_540ad8.csv').drop_duplicates(['condition', 'ep'])


# ---------------------------------------------------------------- Fig. 8: robustness
def fig8():
    e = ep6(); e = e[e.h == 0.2]
    t1 = pd.read_csv(f'{O}/T1_faults_episodes.csv')
    conds = [('Nominal\n(1% loss)', None, None), ('10%\nloss', '10% loss', '10% loss'), ('30%\nloss', '30% loss', '30% loss'),
             ('+300 msec\ndelay', '+300 ms delay', '+300 ms delay'), ('+500 msec\ndelay', '+500 ms delay', '+500 ms delay'),
             ('RSU\nfailure', 'RSU down', 'RSU down')]
    sw, nom = llm_c_sweep(), llm_c_nominal()
    nov2x = np.mean([rate(e[(e.policy == 'PPO, no V2X') & (e.run == r)]) for r in range(5)])
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    w = 0.26
    for i, (lab, c_rl, c_llm) in enumerate(conds):
        if c_rl is None:
            runs = [rate(e[(e.policy == 'PPO + V2X intention') & (e.run == r)]) for r in range(5)]
            rule = e[e.policy == RULE]; llm = nom
        else:
            d = t1[t1.condition == c_rl]
            runs = [rate(d[(d.policy == 'PPO + V2X intention') & (d.run == r)]) for r in range(5)]
            rule = d[d.policy.str.startswith('Rule')]; llm = sw[sw.condition == c_llm]
        lo, hi = tci(runs); m = np.mean(runs)
        ax.bar(i - w, m, w, color=RED, label='RL agent + V2X (5 runs; dots: runs)' if i == 0 else None, zorder=2)
        ax.errorbar(i - w, m, yerr=[[m - lo], [hi - m]], color='k', lw=0.8, capsize=1.5, zorder=3)
        ax.scatter(np.full(5, i - w) + np.linspace(-0.07, 0.07, 5), runs, s=5, color='k', zorder=4)
        for xo, d_, col, lb in ((0, rule, DGRAY, r'Rule, $\tau^\star$ = 0.05'), (w, llm, BLUE, 'LLM, eight examples (C)')):
            k, n = int(d_.collision.sum()), len(d_); m2 = 100 * k / n; l2, h2 = cp(k, n)
            ax.bar(i + xo, m2, w, color=col, label=lb if i == 0 else None, zorder=2)
            ax.errorbar(i + xo, m2, yerr=[[m2 - l2], [h2 - m2]], color='k', lw=0.8, capsize=1.5, zorder=3)
    ax.axhline(nov2x, color='k', ls='--', lw=0.8, zorder=1)
    ax.text(-0.45, nov2x + 0.08, f'RL without V2X, {nov2x:.2f}%', fontsize=6)
    ax.set_xticks(range(len(conds))); ax.set_xticklabels([c[0] for c in conds], fontsize=6)
    ax.set_ylabel('Collisions (%)'); ax.set_ylim(0, None); ax.grid(axis='y', alpha=0.3, lw=0.5)
    ax.legend(loc='upper left', frameon=False, fontsize=6)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    save(fig, 'fig_robustness')


# ---------------------------------------------------------------- Fig. 7 (S4): hesitation, and S3: time under hesitation
def hes_groups():
    e = ep6(); sw, nom = llm_c_sweep(), llm_c_nominal()
    H = [0.0, 0.2, 0.4, 0.6, 0.8]
    llm = {0.0: sw[sw.condition == 'h = 0.0'], 0.2: nom, 0.8: sw[sw.condition == 'h = 0.8']}
    G = [('Floor\nonly', LGRAY, {h: e[(e.policy == FLOOR) & (e.h == h)] for h in H}, None),
         ('RL,\nno V2X', PINK, {h: e[(e.policy == 'PPO, no V2X') & (e.h == h)] for h in H}, 5),
         ('RL +\nV2X', RED, {h: e[(e.policy == 'PPO + V2X intention') & (e.h == h)] for h in H}, 5),
         ('Rule,\n' + r'$\tau^\star$ = 0.05', DGRAY, {h: e[(e.policy == RULE) & (e.h == h)] for h in H}, None),
         ('Oracle\nrule', ORANGE, {h: e[(e.policy == ORACLE) & (e.h == h)] for h in H}, None),
         ('LLM,\nexamples (C)', BLUE, llm, None)]
    return G


def fig7():
    G = hes_groups()
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    ax.axvspan(-0.5, 1.5, color='0.93', zorder=0); ax.axvline(1.5, color='k', ls=':', lw=0.8)
    ax.text(0.5, 2.38, 'Without the intention', ha='center', fontsize=6, style='italic')
    ax.text(3.5, 2.38, 'With the shared intention', ha='center', fontsize=6, style='italic')
    for i, (lab, col, by_h, runs) in enumerate(G):
        hs = sorted(by_h)
        if runs:
            per_run = [np.mean([rate(by_h[h][by_h[h].run == r]) for h in hs]) for r in range(runs)]
            m = np.mean(per_run); lo, hi = tci(per_run)
            dots = [np.mean([rate(by_h[h][by_h[h].run == r]) for r in range(runs)]) for h in hs]
        else:
            k = sum(int(by_h[h].collision.sum()) for h in hs); n = sum(len(by_h[h]) for h in hs)
            m = 100 * k / n; lo, hi = cp(k, n); dots = [rate(by_h[h]) for h in hs]
        ax.bar(i, m, 0.62, color=col, zorder=2)
        ax.errorbar(i, m, yerr=[[m - lo], [hi - m]], color='k', lw=0.8, capsize=2, zorder=3)
        xs = i + (np.array(hs) - 0.4) * 0.6
        ax.scatter(xs, dots, s=4, color='k', zorder=4)
    ax.text(5.0, 1.93, 'dots: $h$ =', ha='center', fontsize=6)
    for j, h in enumerate([0.0, 0.2, 0.4, 0.6, 0.8]):
        ax.text(4.4 + 0.3 * j, 1.66, f'{h}', ha='center', fontsize=4.5); ax.scatter(4.4 + 0.3 * j, 1.8, s=4, color='k')
    ax.set_xticks(range(len(G))); ax.set_xticklabels([g[0] for g in G], fontsize=6)
    ax.set_ylabel('Collisions (%)'); ax.set_ylim(0, 2.5); ax.set_xlim(-0.5, len(G) - 0.5)
    ax.grid(axis='y', alpha=0.3, lw=0.5)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    save(fig, 'fig_hesitation')


def figS3():
    G = hes_groups()
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    for lab, col, by_h, runs in G:
        hs = sorted(by_h)
        y = [by_h[h][by_h[h].passed].time.mean() for h in hs]
        ax.plot(hs, y, '-o', color=col, ms=3, lw=1, label=lab.replace('\n', ' '))
    ax.set_xlabel('Hesitation level $h$'); ax.set_ylabel('Mean time to pass (sec)'); ax.grid(alpha=0.3, lw=0.5)
    ax.legend(fontsize=6, ncol=2)
    save(fig, 'figS3_hesitation_time')


# ---------------------------------------------------------------- Fig. 4 and S1: training
def train_curves(files, key):
    t = pd.read_csv(f'{D}/training_logs_v6.csv')
    out = []
    for f in files:
        d = t[(t.file == f) & (t.kind == ('val' if key == 'val' else 'train'))].sort_values('frames')
        out.append((d.frames.values, d.ret.values if key in ('val', 'ret') else 100 * d.coll.values))
    n = min(len(o[0]) for o in out)
    x = out[0][0][:n]; Y = np.array([o[1][:n] for o in out])
    return x, Y


GROUPS4 = [('RL, no V2X (5 runs)', [f'PPOnoV2X_run{r}.pt' for r in range(5)], PINK, '-'),
           ('RL + V2X, random weights (2 runs)', [f'PPOV2Xintentionfromscratch_run{r}.pt' for r in range(2)], RED, ':'),
           ('RL + V2X, PPO after imitation (5 runs)', [f'PPOV2Xintention_run{r}.pt' for r in range(5)], '0.3', '-')]


def ema(y, a=0.1):
    out, m = [], None
    for v in y:
        m = v if m is None else a * v + (1 - a) * m
        out.append(m)
    return np.array(out)


def fig4():
    """Fig. S5 (old Fig. 4): (a) training collisions, EMA-smoothed; (b) validation return every 81,920 decisions.
    Bands: 95% t-intervals over the five runs; the two runs from random weights by their mean only."""
    t = pd.read_csv(f'{D}/training_logs_v6.csv')
    imit = t[(t.kind == 'chosen') & t.file.str.startswith('PPOV2Xintention_run')].ret.mean()
    for key, name, ylab in (('coll', 'fig_train_a_collisions', 'Collisions (%)'), ('val', 'fig_train_b_validation', 'Validation return')):
        fig, ax = plt.subplots(figsize=(1.7, 1.25))
        for lab, files, col, ls in GROUPS4:
            x, Y = train_curves(files, key)
            if key == 'coll':
                Y = np.array([ema(y) for y in Y])
            if key == 'val' and files[0].startswith('PPOV2Xintention_run'):
                x = np.concatenate([[0], x]); Y = np.column_stack([np.full(len(Y), imit), Y])
            m = Y.mean(0)
            ax.plot(x / 1e5, m, color=col, ls=ls, lw=0.9)
            if len(files) >= 5:
                h = stats.t.ppf(0.975, len(files) - 1) * Y.std(0, ddof=1) / math.sqrt(len(files))
                ax.fill_between(x / 1e5, m - h, m + h, color=col, alpha=0.18, lw=0)
        if key == 'val':
            ax.axhline(imit, color='k', ls='--', lw=0.7); ax.text(0.3, imit + 0.4, f'imitation network ({imit:.2f})', fontsize=5.5)
        ax.set_xlabel(r'Decisions ($\times 10^5$)'); ax.set_ylabel(ylab); ax.grid(alpha=0.3, lw=0.4); ax.set_xlim(0, 12)
        for s_ in ('top', 'right'): ax.spines[s_].set_visible(False)
        save(fig, name)
    fig, ax = plt.subplots(figsize=(3.5, 0.35)); ax.axis('off')
    for lab, files, col, ls in GROUPS4:
        ax.plot([], [], color=col, ls=ls, lw=1, label=lab)
    ax.legend(loc='center', ncol=2, frameon=False, fontsize=5.5); save(fig, 'fig_train_legend')


def figS1():
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    for lab, files, col, ls in GROUPS4:
        x, Y = train_curves(files, 'ret')
        ax.plot(x / 1e5, Y.mean(0), color=col, ls=ls, lw=1, label=lab)
        ax.fill_between(x / 1e5, Y.min(0), Y.max(0), color=col, alpha=0.15, lw=0)
    ax.set_xlabel(r'Decisions ($\times 10^5$)'); ax.set_ylabel('Training return (last 200 episodes)')
    ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=6)
    save(fig, 'figS1_training_return')


# ---------------------------------------------------------------- Fig. 5: predictor
def fig5():
    p_ = f'{O}/T8_predictor_test_windows.csv'
    if not os.path.exists(p_):
        print('run rerun_tests.py T8 first'); return
    d = pd.read_csv(p_)
    rng = np.random.default_rng(0); eps = d.ep.unique()
    boots = [rng.choice(eps, len(eps)) for _ in range(1000)]
    d['bin'] = pd.cut(d.p, [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0001], right=False, labels=False)
    groups = {e: g for e, g in d.groupby('ep')}
    # (a) calibration
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    xs, ys, lo, hi, ns = [], [], [], [], []
    for b, g in d.groupby('bin'):
        xs.append(g.p.mean()); ys.append(g.y.mean()); ns.append(len(g))
    bs = []
    for s in boots:
        db = pd.concat([groups[e] for e in s])
        bs.append(db.groupby('bin').y.mean().reindex(sorted(d.bin.unique())).values)
    bs = np.array(bs); lo, hi = np.nanpercentile(bs, 2.5, 0), np.nanpercentile(bs, 97.5, 0)
    ax.errorbar(xs, ys, yerr=[np.array(ys) - lo, hi - np.array(ys)], fmt='-o', color=BLUE, ms=2.5, lw=0.9, capsize=1.5)
    for x, y, n in zip(xs, ys, ns):
        ax.text(x + 0.012, y - 0.06, f'{n:,}', fontsize=5)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.7); ax.text(0.88, 0.62, 'perfect\ncalibration', fontsize=5.5, ha='center')
    ax.set_xlabel(r'Predicted intention $\hat p$ (bin mean)'); ax.set_ylabel('Share distracted'); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, lw=0.4)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    save(fig, 'fig_pred_a_calibration')
    # (b) AUC by distance to the curb
    from sklearn.metrics import roc_auc_score
    bins = [(0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 5), (5, 10)]
    fig, ax = plt.subplots(figsize=(COL_W, 1.6))
    au, l_, h_ = [], [], []
    for a, b in bins:
        s = d[(d.dist > a) & (d.dist <= b)]
        au.append(roc_auc_score(s.y, s.p))
        sb = []
        for bb in boots[:300]:
            db = pd.concat([groups[e] for e in bb]); db = db[(db.dist > a) & (db.dist <= b)]
            if db.y.nunique() == 2: sb.append(roc_auc_score(db.y, db.p))
        l_.append(np.percentile(sb, 2.5)); h_.append(np.percentile(sb, 97.5))
    x = np.arange(len(bins))
    ax.bar(x, au, 0.65, color=BLUE, zorder=2)
    ax.errorbar(x, au, yerr=[np.array(au) - l_, np.array(h_) - au], fmt='none', color='k', lw=0.8, capsize=1.5, zorder=3)
    ax.axhline(0.5, color='k', ls='--', lw=0.7); ax.text(len(bins) - 1.0, 0.535, 'no information (0.5)', fontsize=5.5, ha='center', zorder=5, bbox=dict(fc='white', ec='none', pad=0.5))
    ax.set_xticks(x); ax.set_xticklabels([f'{a:g} to {b:g}' for a, b in bins]); ax.set_ylim(0, 1.05)
    ax.set_xlabel('Distance of the pedestrian to the curb (m)'); ax.set_ylabel('AUC'); ax.grid(axis='y', alpha=0.3, lw=0.4)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    save(fig, 'fig_pred_b_auc')


# ---------------------------------------------------------------- Fig. 6: when the agents brake
PBINS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.01]
PLABS = ['<0.05', '0.05 to\n0.1', '0.1 to\n0.2', '0.2 to\n0.3', '0.3 to\n0.5', '0.5 to\n0.7', r'$\geq$0.7']
STOP_MARGIN, A_BRAKE, DT3 = 1.4, 2.8, 0.3


def braking_window(d):
    D_ = d.speed ** 2 / (2 * A_BRAKE) + d.speed * DT3
    return d.conflict.astype(bool) & ((d.gap - STOP_MARGIN) <= D_ + 2.0) & (d.gap > STOP_MARGIN)


def fig6_table():
    rl = pd.read_csv(f'{O}/T7_decisions_rl_rule.csv')
    rows = [('Reference\nrule, ' + r'$\tau^\star$ = 0.05', rl[rl.agent.str.startswith('Rule')], 'action'),
            ('RL + V2X, 5 runs', rl[rl.agent == 'RL + V2X'], 'action')]
    for lab, f, cond in (('Zero-shot', 'decisions_h0.2.csv', 'LLM + V2X, instant answers'), ('A, cost hint', 'decisions_ablation_de3259_h0.2.csv', None),
                         ('B, threshold', 'decisions_ablation_dd34e6_h0.2.csv', None), ('C, eight examples', 'decisions_ablation_540ad8_h0.2.csv', None),
                         ("C$'$, second set", 'decisions_ablation_1df929_h0.2.csv', None)):
        x = pd.read_csv(f'{D}/{f}')
        if cond: x = x[x.condition == cond]
        rows.append((lab, x, 'action'))
    out = {}
    for panel, where in (('sidewalk', lambda d: d.ped_from_road > 0), ('road', lambda d: (d.ped_from_road <= 0) & (d.ped_from_road >= -6))):
        M, Nn = [], []
        for lab, x, col in rows:
            s = x[braking_window(x) & where(x)]
            b = pd.cut(s.p_used, PBINS, right=False, labels=False)
            M.append([100 * (s[b == k][col] == 2).mean() if (b == k).any() else np.nan for k in range(len(PLABS))])
            Nn.append([int((b == k).sum()) for k in range(len(PLABS))])
        out[panel] = ([r[0] for r in rows], np.array(M), np.array(Nn))
    return out


def fig6():
    T = fig6_table()
    for panel, name in (('sidewalk', 'fig_braking_a_sidewalk'), ('road', 'fig_braking_b_road')):
        labs, M, Nn = T[panel]
        fig, ax = plt.subplots(figsize=(COL_W, 2.0))
        im = ax.imshow(M, cmap='Blues', vmin=0, vmax=100, aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isnan(M[i, j]): continue
                ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', fontsize=5.5, color='white' if M[i, j] > 60 else 'k',
                        fontweight='bold' if i in (1, 5) else None)
                if i > 0 and not np.isnan(M[0, j]) and abs(M[i, j] - M[0, j]) <= 10:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=ORANGE, lw=1.2))
        if panel == 'sidewalk':
            ax.axvline(0.5, color='k', ls='--', lw=0.8)
        ax.axhline(0.5, color='k', lw=0.8)
        ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=5.5)
        ax.set_xticks(range(len(PLABS))); ax.set_xticklabels(PLABS, fontsize=5)
        ax.set_xlabel(r'Intention $p$ sent by the RSU over V2X'); ax.set_ylabel('AV decision policy')
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label('Braking (%)', fontsize=6); cb.ax.tick_params(labelsize=5)
        save(fig, name)
        pd.DataFrame(M, index=[l.replace('\n', ' ') for l in labs], columns=[p.replace('\n', ' ') for p in PLABS]).round(1).to_csv(f'{F}/{name}_values.csv')
        pd.DataFrame(Nn, index=[l.replace('\n', ' ') for l in labs], columns=[p.replace('\n', ' ') for p in PLABS]).to_csv(f'{F}/{name}_counts.csv')


# ---------------------------------------------------------------- Fig. 10: one scenario, both agents
def fig10():
    t = pd.read_csv(f'{O}/T5_replay_trace.csv')
    TS = [0.0, 1.5, 3.0, 4.5, 6.0]
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, 2.95), sharex=True, gridspec_kw=dict(hspace=0.42))
    titles = {'RL + V2X, run 4': '(a) RL agent with V2X (run 4)', 'LLM + V2X, variant C': '(b) LLM agent with V2X, eight examples (variant C)'}
    for ax, agent in zip(axes, ['RL + V2X, run 4', 'LLM + V2X, variant C']):
        g = t[t.agent == agent].reset_index(drop=True)
        at = {ts: g.iloc[(g.t - ts).abs().argmin()] for ts in TS}
        ax.set_xlim(31.3, -3.5); ax.set_ylim(-2.5, 3.55); ax.set_yticks([])
        for sp_ in ('top', 'right', 'left'): ax.spines[sp_].set_visible(False)
        ax.add_patch(Rectangle((-10, 0), 45, 3.0, color='0.88', zorder=0))           # road
        ax.add_patch(Rectangle((-10, -2.5), 45, 2.5, color='#f3efe6', zorder=0))      # sidewalk
        ax.plot([35, -10], [0, 0], color='k', lw=1, zorder=1)
        ax.plot([35, 9.0], [1.4, 1.4], color='white', ls='--', lw=0.8, zorder=1)
        ax.plot([12, 12], [0, 3.0], color='k', ls=':', lw=0.8); ax.text(12, 3.12, '$d_o$', ha='center', fontsize=6.5)
        ax.plot([1.4, 1.4], [1.45, 3.0], color='k', lw=1.4); ax.text(5.3, 3.12, 'stop line', fontsize=6, ha='center')
        ax.plot([0, 0], [-2.5, 3.0], color='0.35', ls=':', lw=0.7); ax.text(-1.6, 3.12, 'crosswalk', fontsize=6, ha='center')
        for x0 in (25.5, 18.5, 11.5, 4.5):                                            # parked cars
            ax.add_patch(FancyBboxPatch((x0, 0.18), 4.0, 0.95, boxstyle='round,pad=0,rounding_size=0.25', fc='0.8', ec='0.6', lw=0.5, zorder=2))
            ax.add_patch(Rectangle((x0 + 1.2, 0.4), 1.0, 0.5, color='0.6', zorder=3))
        groups = []                                                                    # times whose car positions overlap
        for ts in TS:
            if groups and abs(at[groups[-1][-1]].gap - at[ts].gap) < 1.6:
                groups[-1].append(ts)
            else:
                groups.append([ts])
        for grp in groups:
            r = at[grp[-1]]; ts0 = grp[0]
            alpha = 0.45 if ts0 < 1.5 else (0.6 if ts0 < 3 else (0.8 if ts0 < 4.5 else 1.0))
            ax.add_patch(FancyBboxPatch((r.gap, 1.6), 4.0, 1.0, boxstyle='round,pad=0,rounding_size=0.2', fc=RED, ec='0.35', alpha=alpha, lw=0.5, zorder=4))
            ax.add_patch(Rectangle((r.gap + 1.2, 1.85), 1.0, 0.5, color='#7a1010', alpha=alpha, zorder=5))
            braking = any(int(at[x].action) == 2 or bool(at[x].floor_brake) for x in grp)
            if braking:
                for yy in (1.58, 2.5):
                    ax.add_patch(Rectangle((r.gap - 0.15, yy), 0.4, 0.13, color='#ffd400', zorder=6))
            ax.text(r.gap + 2.0, 2.78, ', '.join(f'{x:g}' for x in grp), ha='center', fontsize=6)
        for ts in TS:                                                                  # pedestrian, every 1.5 sec
            r = at[ts]
            yy = 0.25 + (48.3 - r.xp) / 6.3 * 2.5 if r.xp <= 48.3 else -0.35 - (r.xp - 48.3) / 3.0 * 1.6
            ax.scatter([0], [yy], s=26, marker='o', facecolor='#2b3a55' if r.visible else 'white', edgecolor='#2b3a55',
                       linestyle='-' if r.visible else ':', linewidth=0.8, zorder=7)
            ax.text(-0.8, yy, f'{ts:g}', fontsize=6, va='center')
        b = g[g.action == 2].iloc[0]; stop = g[g.sv < 0.05]; t_end = g.t.max() + 0.1
        ax.text(30.6, -0.8, f'Brakes at {b.t:.1f} sec on $p$ = {b.p_used:.2f}, with the pedestrian {"hidden" if not b.visible else "visible"}.', fontsize=5.8, zorder=8)
        line2 = (f'Stops at {stop.t.min():.1f} sec, waits, and passes at {t_end:.1f} sec.' if len(stop) and stop.gap.iloc[0] > 8
                 else f'Waits for the pedestrian and passes at {t_end:.1f} sec.')
        ax.text(30.6, -1.75, line2, fontsize=5.8, zorder=8)
        ax.add_patch(FancyBboxPatch((1.7, -1.55), 3.0, 0.8, boxstyle='round,pad=0', fc='#4a7fd4', ec='#2a5fb4', lw=0.6, zorder=8))
        ax.text(3.2, -1.15, 'RSU', color='white', fontsize=5.5, ha='center', va='center', zorder=9, fontweight='bold')
        rb = at[1.5]
        ax.plot([3.2, rb.gap + 2.0], [-0.75, 2.1], color='#4a7fd4', ls='--', lw=0.8, zorder=3)
        ax.set_title(titles[agent], fontsize=7, fontweight='bold', loc='left', pad=9)
    axes[1].set_xlabel('Distance to the crosswalk (m)'); axes[1].set_xticks([25, 20, 15, 10, 5, 0])
    save(fig, 'fig_run_tracked')


ALL = ['fig4', 'fig5', 'fig6', 'fig7', 'fig8', 'fig10', 'figS1', 'figS3']
if __name__ == '__main__':
    for name in (sys.argv[1:] or ALL):
        try:
            globals()[name]()
        except FileNotFoundError as ex:
            print(name, 'skipped, missing input:', ex.filename)
