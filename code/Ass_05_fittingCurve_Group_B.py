import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from sklearn.metrics import r2_score, mean_squared_error

# Data is not redistributed with this academic repository. Set
# HYDROLOGY_RATING_CURVE_PATH to the authorized semicolon-separated source file.
csv_path = Path(os.environ.get(
    "HYDROLOGY_RATING_CURVE_PATH",
    Path(__file__).resolve().parent / "data" / "time_series___24163005_without_Outliers.csv",
))
if not csv_path.is_file():
    raise FileNotFoundError(
        "Rating-curve data was not found. Set HYDROLOGY_RATING_CURVE_PATH to "
        "an authorized time_series___24163005_without_Outliers.csv file."
    )
df = pd.read_csv(csv_path, sep=';')

h = df['ddho__ref'].to_numpy(dtype=float)
Q = df['diso__ref'].to_numpy(dtype=float)

valid_mask = (h > 0) & (Q > 0)
h = h[valid_mask]
Q = Q[valid_mask]

#take median as h instead of repeating calculating
USE_GROUPED = False
if USE_GROUPED:
    tmp = pd.DataFrame({'h': h, 'Q': Q})
    tmp = tmp.groupby('h', as_index=False)['Q'].median()
    h = tmp['h'].to_numpy(dtype=float)
    Q = tmp['Q'].to_numpy(dtype=float)


# 1. Define the fitting functions
def power_model(h, a, b, c):
    x = np.maximum(h - b, 1e-8)
    return a * np.power(x, c)

def sigmoid_weight(h, h0, s):
    z = (h - h0) / s
    z = np.clip(z, -60, 60)
    return 1.0 / (1.0 + np.exp(-z))

def smooth_two_power(h, a1, b1, c1, a2, b2, c2, h0, s):
    q1 = power_model(h, a1, b1, c1)
    q2 = power_model(h, a2, b2, c2)
    w = sigmoid_weight(h, h0, s)
    return (1.0 - w) * q1 + w * q2

# 2. finding best power during interval from suggestion
def find_best_shifted_power(h_data, Q_data, c_min=0.5, c_max=6.0, c_step=0.02):
    best_r2 = -np.inf
    best_params = None

    for c in np.arange(c_min, c_max + c_step, c_step):
        Y = Q_data ** (1.0 / c)
        X = h_data

        slope, intercept = np.polyfit(X, Y, 1)
        if slope <= 0:
            continue

        a_est = slope ** c
        b_est = -intercept / slope

        if np.any(h_data <= b_est):
            continue

        Q_pred = power_model(h_data, a_est, b_est, c)
        r2 = r2_score(Q_data, Q_pred)

        if r2 > best_r2:
            best_r2 = r2
            best_params = (a_est, b_est, c)

    return best_params, best_r2


# 3. First a trial for a global power function
global_params, global_r2 = find_best_shifted_power(h, Q, c_min=0.5, c_max=6.0, c_step=0.01)
if global_params is None:
    raise RuntimeError("Global power fit failed.")

a_g, b_g, c_g = global_params
Q_global_pred = power_model(h, a_g, b_g, c_g)
global_rmse = np.sqrt(mean_squared_error(Q, Q_global_pred))


# 4. Finding Bipower smoothing model of inital values h_0
left_mask = h <= 442
right_mask = h >= 447

left_init, _ = find_best_shifted_power(h[left_mask], Q[left_mask], c_min=0.5, c_max=5.0, c_step=0.02)
right_init, _ = find_best_shifted_power(h[right_mask], Q[right_mask], c_min=0.5, c_max=5.0, c_step=0.02)

if left_init is None or right_init is None:
    raise RuntimeError("Initial left/right power fit failed.")

a1_0, b1_0, c1_0 = left_init
a2_0, b2_0, c2_0 = right_init

# Initial transition centre and transition width
h0_0 = 446.0
s_0 = 2.0

p0 = [a1_0, b1_0, c1_0, a2_0, b2_0, c2_0, h0_0, s_0]

# bounds for parameters
lb = [
    1e-12,   -100.0, 0.5,   # a1, b1, c1
    1e-12,   300.0,  0.5,   # a2, b2, c2
    440.0,   0.3            # h0, s
]

ub = [
    1.0,     100.0,  5.0,   # a1, b1, c1
    10.0,    440.0,  6.0,   # a2, b2, c2
    450.0,   8.0            # h0, s
]


# 5. least_squares for fitting also with residuals plotting to show
def residuals(params, h_data, Q_data):
    pred = smooth_two_power(h_data, *params)


    local_weight = 1.0 + 3.0 * np.exp(-0.5 * ((h_data - 446.0) / 3.0) ** 2)

    return np.sqrt(local_weight) * (pred - Q_data)

result = least_squares(
    residuals,
    x0=p0,
    bounds=(lb, ub),
    args=(h, Q),
    max_nfev=50000
)

if not result.success:
    print("Optimization message:", result.message)

popt = result.x
a1, b1, c1, a2, b2, c2, h0, s = popt

Q_smooth_pred = smooth_two_power(h, *popt)
smooth_r2 = r2_score(Q, Q_smooth_pred)
smooth_rmse = np.sqrt(mean_squared_error(Q, Q_smooth_pred))


# 6. results
print("=" * 72)
print("Single global power fit")
print("=" * 72)
print(f"Q = {a_g:.8e} * (h - {b_g:.6f})^{c_g:.4f}")
print(f"R²   = {global_r2:.6f}")
print(f"RMSE = {global_rmse:.6f}")

print("\n" + "=" * 72)
print("Smooth two-power fit (sigmoid-weighted)")
print("=" * 72)
print(f"Q1(h) = {a1:.8e} * (h - {b1:.6f})^{c1:.4f}")
print(f"Q2(h) = {a2:.8e} * (h - {b2:.6f})^{c2:.4f}")
print(f"w(h)  = 1 / (1 + exp(-(h - {h0:.4f}) / {s:.4f}))")
print(f"Q(h)  = (1-w) * Q1 + w * Q2")

print(f"\nFinal R²   = {smooth_r2:.6f}")
print(f"Final RMSE = {smooth_rmse:.6f}")

print("\nCompared with single global power:")
print(f"ΔR²   = {smooth_r2 - global_r2:.6f}")
print(f"RMSE improvement = {global_rmse - smooth_rmse:.6f}")


# 7. plotting
plt.figure(figsize=(10, 6))
plt.scatter(h, Q, color='gray', alpha=0.25, s=10, label='Original Data')

x_line = np.linspace(h.min(), h.max(), 500)

# global power(results below is not okay so we'd better take the second one)
y_global = power_model(x_line, a_g, b_g, c_g)
plt.plot(x_line, y_global, '--', linewidth=2, label=f'Global Power (R²={global_r2:.3f})')

# 平滑双幂律
y_smooth = smooth_two_power(x_line, *popt)
plt.plot(x_line, y_smooth, '-', linewidth=2.5, label=f'Smooth Two-Power (R²={smooth_r2:.3f})')

plt.axvline(h0, linestyle=':', linewidth=1.5, label=f'Transition center h0={h0:.1f}')

plt.xlabel('h (ddho__ref)')
plt.ylabel('Q (diso__ref)')
plt.title('Smooth Two-Power Fit')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# 8. residuals plotting(not important just for math performance determination )
res_global = Q - Q_global_pred
res_smooth = Q - Q_smooth_pred

plt.figure(figsize=(10, 5))
plt.scatter(h, res_global, s=10, alpha=0.25, label='Global Power Residual')
plt.scatter(h, res_smooth, s=10, alpha=0.25, label='Smooth Two-Power Residual')
plt.axhline(0, linestyle='--', linewidth=1)
plt.axvline(h0, linestyle=':', linewidth=1.5, label=f'h0={h0:.1f}')
plt.xlabel('h (ddho__ref)')
plt.ylabel('Residual = Q_obs - Q_fit')
plt.title('Residual Comparison')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
