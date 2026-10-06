"""
Fundamental Monte Carlo for the Fabrinet (FN) long. Page 1 of the memo.

One path:  regime -> FY28 revenue -> margin -> EPS -> P/E (rises with growth) -> price -> return.
Revenue feeds both EPS and the P/E, so bad paths compound. That is the fat left tail.

Run: python montecarlo.py   -> prints the memo numbers, writes mc_returns.pdf/.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

PRICE_TODAY   = 453.84           # Oct 5, 2026 close
SHARES        = 149.1 / 4.10     # 36.37M diluted: FQ4 non-GAAP net income / EPS
BASE_REVENUE  = 8228.0           # $M, our FY28 revenue
STREET_FY27   = 6100.0           # $M, growth is measured against this
P_BEAR        = 0.25             # probability of a capex pause. The key judgment.
BEAR_REV_MEAN, BEAR_REV_SD = 6300.0, 450.0
BASE_REV_SD   = 0.10             # lognormal sigma around the base case
MARGIN_MEAN, MARGIN_SD, MARGIN_BEAR = 0.108, 0.005, 0.006
MARGIN_CLIP   = (0.08, 0.13)
PE_INTERCEPT, PE_SLOPE, PE_NOISE_SD = 14.05, 0.243, 2.5   # P/E = a + b * growth% + noise
PE_CLIP       = (9.0, 34.0)
N_PATHS, SEED = 200_000, 11


def simulate(n=N_PATHS, seed=SEED, p_bear=P_BEAR, pe_intercept=PE_INTERCEPT,
             pe_slope=PE_SLOPE, margin_mean=MARGIN_MEAN, net_cash=0.0):
    """n simulated 12-month returns."""
    rng = np.random.default_rng(seed)
    bear = rng.random(n) < p_bear
    revenue = np.where(bear, rng.normal(BEAR_REV_MEAN, BEAR_REV_SD, n),
                       BASE_REVENUE * np.exp(rng.normal(0, BASE_REV_SD, n)))
    margin = np.clip(rng.normal(margin_mean, MARGIN_SD, n) - bear * MARGIN_BEAR, *MARGIN_CLIP)
    eps = revenue * margin / SHARES
    growth = (revenue / STREET_FY27 - 1) * 100
    pe = np.clip(pe_intercept + pe_slope * growth + rng.normal(0, PE_NOISE_SD, n), *PE_CLIP)
    return (eps * pe + net_cash) / PRICE_TODAY - 1


def cvar5(r):
    return r[r <= np.percentile(r, 5)].mean()


def kelly(r, rf=0.04):
    """Weight f that maximises E[log(1 + f*r + (1-f)*rf)]."""
    fs = np.linspace(0, 1, 201)
    return fs[np.argmax([np.mean(np.log1p(f * r + (1 - f) * rf)) for f in fs])]


def plot(r, path="mc_returns"):
    INK, SECOND, MUTED, AXIS, GAIN, LOSS = "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#c2c7cc", "#d03b3b"
    median = np.median(r)
    counts, edges = np.histogram(r, bins=np.arange(-0.76, 1.68, 0.04))
    share = counts / len(r) * 100
    x = (edges[:-1] + edges[1:]) / 2
    top = share.max()
    fig, ax = plt.subplots(figsize=(3.6, 1.95))
    ax.fill_between(x, share, step="mid", where=x < 0, color=LOSS, lw=0)
    ax.fill_between(x, share, step="mid", where=x >= 0, color=GAIN, lw=0)
    ax.plot([0, 0], [0, top * 1.07], color=INK, lw=0.9)
    ax.text(0, top * 1.10, "today", fontsize=6, color=SECOND, ha="center")
    ax.text(-0.33, top * 0.34, f"{(r < 0).mean():.0%}\nlose money", fontsize=7.2, color="white",
            ha="center", va="center", weight="bold")
    ax.text(max(median, 0) + 0.06, top * 0.30, f"median\n{median:+.0%}", fontsize=7.2, color=INK, va="center")
    ax.set_title("Simulated 12-month return, 200,000 paths", fontsize=7.5, color=INK, loc="left",
                 pad=15, weight="bold")
    ax.text(0, 1.035, f"Worst 5% of paths average {cvar5(r):.0%}".replace("-", "−"),
            transform=ax.transAxes, fontsize=6.5, color=SECOND)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xticks([-0.5, 0, 0.5, 1.0, 1.5]); ax.set_xlim(-0.82, 1.66)
    ax.set_ylim(0, top * 1.24); ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=6, colors=MUTED, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    fig.tight_layout(pad=0.25)
    fig.savefig(f"{path}.pdf"); fig.savefig(f"{path}.png", dpi=240)


if __name__ == "__main__":
    r = simulate()
    print(f"mean {r.mean():+.1%}  median {np.median(r):+.1%}  P(loss) {(r < 0).mean():.1%}  "
          f"worst-5% avg {cvar5(r):+.1%}  full Kelly {kelly(r):.2f}")
    print("\nIf someone attacks an input:      mean   P(loss)  worst-5%")
    for label, kw in [("bear case 35% likely", dict(p_bear=0.35)),
                      ("bear case 40% likely", dict(p_bear=0.40)),
                      ("net margin 10.0%, not 10.8%", dict(margin_mean=0.100)),
                      ("harsher P/E rule (10 + 0.30g)", dict(pe_intercept=10.0, pe_slope=0.30)),
                      ("credit $30/share of cash", dict(net_cash=30.0))]:
        s = simulate(**kw)
        print(f"  {label:<30}{s.mean():>+7.1%}{(s < 0).mean():>9.1%}{cvar5(s):>10.1%}")
    plot(r)
