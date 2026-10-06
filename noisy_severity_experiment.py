import numpy as np
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import time

DT = 0.001

def _recruitment(amp_ma, pw_ms):
    charge = amp_ma * (pw_ms / 0.5)
    return 1.0 / (1.0 + np.exp(-(charge - 4.0)))

def _frequency_effect(freq_hz):
    return np.tanh(freq_hz / 30.0)

def simulate_circuit(freq_hz, amp_ma, pw_ms, injury_severity, duration_s=8.0, seed=0):
    rng = np.random.default_rng(seed)
    n_steps = int(duration_s / DT)
    t = np.arange(n_steps) * DT
    ees_effect = _recruitment(amp_ma, pw_ms) * _frequency_effect(freq_hz)
    descending_effect = 0.9 * (1.0 - injury_severity)
    u = ees_effect + descending_effect
    tau_x = 0.10
    tau_v = 0.50
    b = 2.5
    w = 2.0
    x1 = np.zeros(n_steps); x2 = np.zeros(n_steps)
    v1 = np.zeros(n_steps); v2 = np.zeros(n_steps)
    noise = rng.normal(0, 0.02, size=(n_steps, 2))
    for i in range(1, n_steps):
        y1 = max(x1[i-1], 0.0)
        y2 = max(x2[i-1], 0.0)
        dx1 = (-x1[i-1] - w*y2 - b*v1[i-1] + u + noise[i, 0]) / tau_x
        dx2 = (-x2[i-1] - w*y1 - b*v2[i-1] + u + noise[i, 1]) / tau_x
        dv1 = (-v1[i-1] + y1) / tau_v
        dv2 = (-v2[i-1] + y2) / tau_v
        x1[i] = x1[i-1] + DT*dx1
        x2[i] = x2[i-1] + DT*dx2
        v1[i] = v1[i-1] + DT*dv1
        v2[i] = v2[i-1] + DT*dv2
    flexor_rate = np.maximum(x1, 0.0)
    extensor_rate = np.maximum(x2, 0.0)
    twitch_tau = 0.04
    kernel_t = np.arange(0, 0.3, DT)
    kernel = kernel_t / twitch_tau * np.exp(-kernel_t / twitch_tau)
    kernel /= kernel.sum()
    flexor_emg = np.convolve(flexor_rate, kernel, mode='same')
    extensor_emg = np.convolve(extensor_rate, kernel, mode='same')
    return t, flexor_emg, extensor_emg

def gait_quality_score(t, flexor_emg, extensor_emg, settle_s=2.0):
    mask = t > settle_s
    f = flexor_emg[mask]
    e = extensor_emg[mask]
    if np.std(f) < 1e-6 or np.std(e) < 1e-6:
        alternation = 0.0
    else:
        corr = np.corrcoef(f, e)[0, 1]
        alternation = np.clip(-corr, 0, 1)
    amp = np.clip((np.mean(f) + np.mean(e)) / 2 / 0.6, 0, 1)
    sig = f - f.mean()
    if np.std(sig) < 1e-6:
        rhythmicity = 0.0
    else:
        freqs = np.fft.rfftfreq(len(sig), DT)
        power = np.abs(np.fft.rfft(sig)) ** 2
        total_power = power.sum() + 1e-9
        band = (freqs >= 0.3) & (freqs <= 2.0)
        rhythmicity = power[band].sum() / total_power
    score = 0.45 * alternation + 0.30 * amp + 0.25 * rhythmicity
    return float(np.clip(score, 0, 1))

def evaluate_params(freq_hz, amp_ma, pw_ms, injury_severity, seed=0):
    t, f_emg, e_emg = simulate_circuit(freq_hz, amp_ma, pw_ms, injury_severity, seed=seed)
    return gait_quality_score(t, f_emg, e_emg)

PARAM_BOUNDS = {"freq_hz": (5, 100), "amp_ma": (0, 10), "pw_ms": (0.1, 1.0)}
PARAM_NAMES = list(PARAM_BOUNDS.keys())

def random_params(rng):
    return np.array([rng.uniform(*PARAM_BOUNDS[p]) for p in PARAM_NAMES])

def build_digital_twin_library(n_virtual_patients=30, n_samples_per_patient=20, seed=1):
    rng = np.random.default_rng(seed)
    X, y = [], []
    severities = rng.uniform(0.2, 0.95, n_virtual_patients)
    for sev in severities:
        for _ in range(n_samples_per_patient):
            params = random_params(rng)
            score = evaluate_params(*params, injury_severity=sev, seed=rng.integers(1e6))
            X.append([sev, *params])
            y.append(score)
    return np.array(X), np.array(y)

def fit_prior_gp(X_lib, y_lib):
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(X_lib.shape[1]), nu=2.5) \
             + WhiteKernel(noise_level=0.01)
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0)
    gp.fit(X_lib, y_lib)
    return gp

def naive_search_trials(injury_severity, n_trials, seed):
    rng = np.random.default_rng(seed)
    best = 0.0
    best_so_far = []
    for _ in range(n_trials):
        params = random_params(rng)
        score = evaluate_params(*params, injury_severity=injury_severity, seed=rng.integers(1e6))
        best = max(best, score)
        best_so_far.append(best)
    return np.array(best_so_far)

def meta_bo_trials(injury_severity, n_trials, seed, gp_prior, X_lib, y_lib, n_candidates=300):
    rng = np.random.default_rng(seed)
    kernel_fixed = gp_prior.kernel_
    patient_X, patient_y = [], []
    best = 0.0
    best_so_far = []
    candidates = np.array([random_params(rng) for _ in range(n_candidates)])
    cand_full = np.hstack([np.full((len(candidates), 1), injury_severity), candidates])
    for _ in range(n_trials):
        if len(patient_y) == 0:
            mu, sigma = gp_prior.predict(cand_full, return_std=True)
        else:
            X_aug = np.vstack([X_lib, np.hstack([np.full((len(patient_X), 1), injury_severity),
                                                   np.array(patient_X)])])
            y_aug = np.concatenate([y_lib, np.array(patient_y)])
            gp_fast = GaussianProcessRegressor(kernel=kernel_fixed, optimizer=None,
                                                normalize_y=True, alpha=1e-6)
            gp_fast.fit(X_aug, y_aug)
            mu, sigma = gp_fast.predict(cand_full, return_std=True)
        ucb = mu + 1.5 * sigma
        next_params = candidates[np.argmax(ucb)]
        score = evaluate_params(*next_params, injury_severity=injury_severity, seed=rng.integers(1e6))
        patient_X.append(next_params)
        patient_y.append(score)
        best = max(best, score)
        best_so_far.append(best)
    return np.array(best_so_far)

def noisy_severity(true_severity, sigma, seed, clip_range=(0.2, 0.95)):
    rng = np.random.default_rng(seed)
    noisy = true_severity + rng.normal(0, sigma)
    return float(np.clip(noisy, *clip_range))

def meta_bo_trials_noisy(true_severity, known_severity, n_trials, seed,
                          gp_prior, X_lib, y_lib, n_candidates=300):

    rng = np.random.default_rng(seed)
    kernel_fixed = gp_prior.kernel_
    patient_X, patient_y = [], []
    best = 0.0
    best_so_far = []
    candidates = np.array([random_params(rng) for _ in range(n_candidates)])
    cand_full = np.hstack([np.full((len(candidates), 1), known_severity), candidates])
    for _ in range(n_trials):
        if len(patient_y) == 0:
            mu, sigma_pred = gp_prior.predict(cand_full, return_std=True)
        else:
            X_aug = np.vstack([X_lib, np.hstack([np.full((len(patient_X), 1), known_severity),
                                                   np.array(patient_X)])])
            y_aug = np.concatenate([y_lib, np.array(patient_y)])
            gp_fast = GaussianProcessRegressor(kernel=kernel_fixed, optimizer=None,
                                                normalize_y=True, alpha=1e-6)
            gp_fast.fit(X_aug, y_aug)
            mu, sigma_pred = gp_fast.predict(cand_full, return_std=True)
        ucb = mu + 1.5 * sigma_pred
        next_params = candidates[np.argmax(ucb)]
        score = evaluate_params(*next_params, injury_severity=true_severity, seed=rng.integers(1e6))
        patient_X.append(next_params)
        patient_y.append(score)
        best = max(best, score)
        best_so_far.append(best)
    return np.array(best_so_far)

t_start = time.time()

print("=" * 60)
print("Building digital twin library (unchanged from original)...")
X_lib, y_lib = build_digital_twin_library(n_virtual_patients=30, n_samples_per_patient=20)
gp = fit_prior_gp(X_lib, y_lib)
print(f"Library built: {X_lib.shape}, elapsed {time.time()-t_start:.1f}s")

n_trials = 15
test_severities = [0.3, 0.6, 0.85]
repeat_seeds = [101, 102, 103, 104, 105, 106]
noise_seed_base = 2024
noise_levels = [0.0, 0.05, 0.15, 0.25]  

print("\nRunning naive search baseline (unchanged)...")
naive_trial1 = {}   
naive_final = {}
for sev in test_severities:
    curves = [naive_search_trials(sev, n_trials, seed=100+r) for r in range(len(repeat_seeds))]
    naive_trial1[sev] = np.array([c[0] for c in curves])
    naive_final[sev] = np.mean([c[-1] for c in curves])
    print(f"  severity {sev}: trial-1 mean={naive_trial1[sev].mean():.4f}, final={naive_final[sev]:.4f}")

print("\nRunning digital-twin search across noise levels...")
bo_trial1 = {}  
for sigma in noise_levels:
    for sev in test_severities:
        curves = []
        for r, base_seed in enumerate(repeat_seeds):
            if sigma == 0.0:
                known_sev = sev
            else:
                known_sev = noisy_severity(sev, sigma, seed=noise_seed_base + r)
            curve = meta_bo_trials_noisy(true_severity=sev, known_severity=known_sev,
                                          n_trials=n_trials, seed=200+base_seed,
                                          gp_prior=gp, X_lib=X_lib, y_lib=y_lib)
            curves.append(curve)
        bo_trial1[(sigma, sev)] = np.array([c[0] for c in curves])
        mean_t1 = bo_trial1[(sigma, sev)].mean()
        print(f"  sigma={sigma:.2f}, severity={sev}: trial-1 mean={mean_t1:.4f}  "
              f"elapsed {time.time()-t_start:.1f}s")

print("\n" + "=" * 60)
print("STATISTICAL COMPARISON (digital-twin vs naive, trial-1 scores)")
print("=" * 60)

summary = []
for sigma in noise_levels:
    dt_scores = np.concatenate([bo_trial1[(sigma, sev)] for sev in test_severities])
    naive_scores = np.concatenate([naive_trial1[sev] for sev in test_severities])
    diffs = dt_scores - naive_scores
    t_stat, p_t = stats.ttest_rel(dt_scores, naive_scores)
    try:
        w_stat, p_w = stats.wilcoxon(dt_scores, naive_scores)
    except ValueError:
        w_stat, p_w = np.nan, np.nan
    n_wins = int(np.sum(dt_scores > naive_scores))
    mean_improvement = diffs.mean()
    significant = (p_t < 0.05) and (p_w < 0.05)
    summary.append((sigma, mean_improvement, n_wins, t_stat, p_t, w_stat, p_w, significant))
    print(f"\nsigma = {sigma}")
    print(f"  digital-twin wins {n_wins}/18 comparisons")
    print(f"  mean improvement: {mean_improvement:.4f}")
    print(f"  paired t-test: t={t_stat:.3f}, p={p_t:.6f}")
    print(f"  Wilcoxon:      W={w_stat}, p={p_w:.6f}")
    print(f"  Statistically significant advantage (alpha=0.05): {significant}")

print("\n" + "=" * 60)
print(f"TOTAL RUNTIME: {time.time()-t_start:.1f} seconds")
print("=" * 60)

np.savez('/home/claude/noisy_severity_results.npz',
          naive_trial1={k: v for k, v in naive_trial1.items()},
          bo_trial1={str(k): v for k, v in bo_trial1.items()},
          summary=np.array(summary, dtype=object))
