"""
Price-path model for Fabrinet (FN), 12 months. Page 2 of the memo.

    S(t+dt) = S(t) * exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z),  Z ~ N(0,1)

GBM has no revenue or margins in it, so it cannot test the thesis. It measures
how far the price moves for no reason, which sets the exit rule, entry price and size.
The view enters only through the drift:
  market: mu = 11% (the reverse-DCF discount rate)   thesis: mu = ln(515 / S0)

Run: python gbm.py   -> prints every number on page 2, writes gbm_page2.pdf/.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import log, sqrt, exp

S0 = 453.84                            # Oct 5, 2026 close
TARGET, BULL, BEAR = 515.0, 721.0, 265.0
SIGMA = 0.65                           # between the 20-day and 49-day figures printed below
MU_MARKET, MU_THESIS = 0.11, log(TARGET / S0)
RF = 0.04                              # assumed return on cash, for Kelly only
STEPS, DT = 252, 1 / 252
DAYS_TO_EARNINGS = 20                  # closes from Oct 6 through the estimated Nov 2 report (after close)
N, SEED = 100_000, 11

# Daily closes Jul 27 to Oct 5, 2026, oldest first. Index 16 is Aug 18 (day after FQ4).
CLOSES = [470.82, 449.72, 414.50, 439.33, 435.41, 455.94, 531.03, 522.22, 543.95,
          562.38, 527.07, 525.88, 571.88, 566.49, 570.22, 598.58, 482.59, 454.55,
          444.87, 436.67, 421.14, 431.54, 437.80, 432.71, 414.36, 412.93, 402.07,
          395.35, 395.00, 407.40, 416.31, 418.27, 403.95, 414.58, 382.55, 374.91,
          392.68, 380.92, 388.55, 401.57, 403.80, 400.85, 398.36, 417.39, 409.96,
          424.12, 424.27, 451.44, 463.69, 453.84]


def simulate(mu, sigma=SIGMA):
    """(N, STEPS+1) price paths. Antithetic pairs halve the noise."""
    z = np.random.default_rng(SEED).standard_normal((N // 2, STEPS)).astype(np.float32)
    z = np.vstack([z, -z])
    steps = (mu - 0.5 * sigma**2) * DT + sigma * sqrt(DT) * z
    return S0 * np.exp(np.concatenate([np.zeros((N, 1), np.float32), np.cumsum(steps, axis=1)], axis=1))


def stats(p, label):
    r = p[:, -1] / S0 - 1
    mdd = (p / np.maximum.accumulate(p, axis=1) - 1).min(axis=1)
    print(f"{label:<8} mean {r.mean():+.1%}  median {np.median(r):+.1%}  P(loss) {(r < 0).mean():.0%}  "
          f"touches $515 {(p.max(axis=1) >= TARGET).mean():.0%}  ends above {(p[:, -1] >= TARGET).mean():.0%}  "
          f"touches $265 {(p.min(axis=1) <= BEAR).mean():.0%}  drawdown>30% {(mdd < -0.30).mean():.0%}  "
          f"median max drawdown {np.median(mdd):.0%}  worst-5% avg {r[r <= np.percentile(r, 5)].mean():+.1%}")
    return r


def plot(p, path="gbm_page2"):
    INK, SECOND, MUTED, AXIS = "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
    BAND, LINE, NAVY, RED = "#cde2fb", "#2a78d6", "#1F3864", "#A61B1B"
    fig, ax = plt.subplots(figsize=(7.4, 2.0))
    t = np.arange(STEPS + 1) / 21
    q = np.percentile(p, [5, 25, 50, 75, 95], axis=0)
    ax.fill_between(t, q[0], q[4], color=BAND, alpha=0.55, lw=0)
    ax.fill_between(t, q[1], q[3], color=BAND, lw=0)
    for i in np.random.default_rng(3).choice(len(p), 3, replace=False):
        ax.plot(t, p[i], color=MUTED, lw=0.5, alpha=0.8)
    ax.plot(t, q[2], color=LINE, lw=1.4)
    for y, lab in [(BULL, "bull $721"), (TARGET, "target $515"), (S0, f"today ${S0:.0f}"), (BEAR, "bear $265")]:
        ax.axhline(y, color=AXIS, lw=0.6, ls=(0, (2, 2)))
        ax.text(12.15, y, lab, fontsize=5.8, color=SECOND, va="center")
    e = DAYS_TO_EARNINGS / 21
    ax.axvline(e, color=NAVY, lw=0.7)
    ax.text(e + 0.12, 985, "FQ1 report", fontsize=5.8, color=NAVY, va="top")
    ax.set_ylim(100, 1000); ax.set_xlim(0, 12); ax.set_xticks([0, 3, 6, 9, 12])
    ax.set_title(f"12-month price paths if the thesis is right (drift {MU_THESIS:.1%}, sigma {SIGMA:.0%})",
                 fontsize=7, loc="left", weight="bold", color=INK)
    ax.set_xlabel("months from today; shaded = middle 50% and 90% of paths", fontsize=6, color=MUTED)
    ax.yaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")

    for a in (ax,):
        a.tick_params(labelsize=6, colors=MUTED, length=0)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            a.spines[s].set_color(AXIS)
    fig.tight_layout(pad=0.3)
    fig.subplots_adjust(right=0.9)
    fig.savefig(f"{path}.pdf"); fig.savefig(f"{path}.png", dpi=240)


if __name__ == "__main__":
    lr = np.diff(np.log(CLOSES))
    vol = lambda x: np.std(x, ddof=1) * sqrt(252)
    print(f"realized vol: 20-day {vol(lr[-20:]):.0%}, 49-day {vol(lr):.0%}, "
          f"49-day without Aug 18 {vol(np.delete(lr, 15)):.0%}; Aug 18 move {exp(lr[15]) - 1:+.1%}")

    thesis = simulate(MU_THESIS)
    r = stats(thesis, "thesis")
    stats(simulate(MU_MARKET), "market")
    assert abs(r.mean() - (exp(MU_THESIS) - 1)) < 0.005, "simulated mean is off the closed form"
    for s in (0.55, 0.75):
        print(f"sigma {s:.0%}: P(loss) {(simulate(MU_THESIS, s)[:, -1] < S0).mean():.0%}")

    sd = SIGMA * sqrt(DAYS_TO_EARNINGS / 252)
    print(f"\nto the report: one sd = {sd:.0%}, ${S0 * exp(-sd):.0f} to ${S0 * exp(sd):.0f}; "
          f"thesis adds {(MU_THESIS - MU_MARKET) * DAYS_TO_EARNINGS / 252:.2%} of drift; "
          f"P(at or above $515 on report day) {(thesis[:, DAYS_TO_EARNINGS] >= TARGET).mean():.0%}")

    print("\nentry  to target  bull/bear  edge vs 11%  Kelly")
    for px in (417, 430, S0, 463.69):
        mu = log(TARGET / px)
        print(f"${px:.0f}   {TARGET / px - 1:+.1%}     {(BULL - px) / (px - BEAR):.1f}x      "
              f"{mu - MU_MARKET:+.1%}      {(mu - RF) / SIGMA**2:.2f}")
    base_bull = (BULL + 2 * 538) / 3          # bull and base kept at 1:2
    print(f"bear probability at which expected value = today's price: {(base_bull - S0) / (base_bull - BEAR):.1%}")
    plot(thesis)
