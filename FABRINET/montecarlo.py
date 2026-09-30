"""
Monte Carlo simulation for the Fabrinet (FN) long thesis.

Draws 200,000 possible versions of FY28, computes the share price implied by
each, and reports the distribution of 12-month returns.

Structure of one path:
    regime  ~ Bernoulli(p_bear)            bad regime or not
    revenue ~ Normal(bear) or Lognormal(base)
    margin  ~ Normal, reduced in the bad regime
    EPS      = revenue * margin / shares
    growth   = revenue vs street FY27 revenue
    P/E     ~ affine in growth, plus noise, clipped
    price    = EPS * P/E + net cash per share
    return   = price / price_today - 1

The link that matters: revenue feeds BOTH earnings and the multiple, so bad
paths compound instead of adding. That is what produces the fat left tail.

Run:  python montecarlo.py
Outputs: summary statistics, a sensitivity table, mc_returns.pdf, mc_returns.png
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ----------------------------------------------------------------------------
# INPUTS  -- every one of these is a judgment you should be able to defend.
#            Sources are in verification_log.md.
# ----------------------------------------------------------------------------
PRICE_TODAY   = 409.96   # $, close of Sep 28 2026 (StockAnalysis / S&P Global)
SHARES        = 149.1 / 4.10   # 36.37M diluted: Q4 non-GAAP net income / EPS
BASE_REVENUE  = 8228.0   # $M, my FY28 revenue forecast
STREET_FY27   = 6100.0   # $M, street FY27 revenue -- the growth baseline
NET_CASH_PS   = 30.0     # $/share of net cash expected by Sep 2027

P_BEAR        = 0.22     # probability of the bad regime. THE key input.
BEAR_REV_MEAN = 6400.0   # $M, revenue in a capex pause
BEAR_REV_SD   = 450.0    # $M
BASE_REV_SD   = 0.10     # lognormal sigma: ~10% spread around the base case

MARGIN_MEAN   = 0.108    # net margin; FY26 ran 10.8-11.3%
MARGIN_SD     = 0.005
MARGIN_BEAR   = 0.004    # margin lost to under-used factories in the bad regime
MARGIN_CLIP   = (0.08, 0.13)

PE_INTERCEPT  = 12.0     # P/E = intercept + slope * growth% + noise
PE_SLOPE      = 0.35
PE_NOISE_SD   = 2.5
PE_CLIP       = (9.0, 34.0)

N_PATHS       = 200_000
SEED          = 11       # fixed so results reproduce exactly


def simulate(n=N_PATHS, seed=SEED, p_bear=P_BEAR, pe_intercept=PE_INTERCEPT,
             pe_slope=PE_SLOPE, margin_mean=MARGIN_MEAN, net_cash=NET_CASH_PS,
             base_rev=BASE_REVENUE, base_rev_sd=BASE_REV_SD,
             bear_rev_mean=BEAR_REV_MEAN):
    """Return an array of n simulated 12-month returns (as fractions)."""
    rng = np.random.default_rng(seed)

    # 1. Which regime is this path in?
    bear = rng.random(n) < p_bear

    # 2. FY28 revenue. Lognormal in the good regime so revenue cannot go
    #    negative and the upside tail is longer than the downside.
    revenue = np.where(
        bear,
        rng.normal(bear_rev_mean, BEAR_REV_SD, n),
        base_rev * np.exp(rng.normal(0, base_rev_sd, n)),
    )

    # 3. Net margin, cut in the bad regime, clipped to a plausible band.
    margin = np.clip(
        rng.normal(margin_mean, MARGIN_SD, n) - bear * MARGIN_BEAR,
        *MARGIN_CLIP,
    )

    # 4. Earnings per share -- pure arithmetic, no randomness.
    eps = revenue * margin / SHARES

    # 5. Growth off the same revenue draw, in percentage points.
    growth = (revenue / STREET_FY27 - 1) * 100

    # 6. The multiple rises with growth. This is the link that fattens the tail.
    pe = np.clip(
        pe_intercept + pe_slope * growth + rng.normal(0, PE_NOISE_SD, n),
        *PE_CLIP,
    )

    # 7. Price, then return.
    price = eps * pe + net_cash
    return price / PRICE_TODAY - 1


def summarise(returns, label="base case"):
    """Print the statistics that actually get quoted in the memo."""
    p5 = np.percentile(returns, 5)
    stats = {
        "mean": returns.mean(),
        "median": np.median(returns),
        "sd": returns.std(),
        "p_loss": (returns < 0).mean(),
        "p_gain_25": (returns > 0.25).mean(),
        "p5": p5,
        "cvar5": returns[returns <= p5].mean(),   # average of the worst 5%
    }
    print(f"\n{label}")
    print(f"  mean return        {stats['mean']:+7.1%}")
    print(f"  median return      {stats['median']:+7.1%}")
    print(f"  spread (1 sd)      {stats['sd']:7.1%}")
    print(f"  chance of a loss   {stats['p_loss']:7.1%}")
    print(f"  chance of >+25%    {stats['p_gain_25']:7.1%}")
    print(f"  5th percentile     {stats['p5']:+7.1%}")
    print(f"  worst 5% average   {stats['cvar5']:+7.1%}   <- this sizes the position")
    return stats


def sensitivity():
    """Re-run under hostile assumptions. Have these numbers ready in Q&A."""
    print("\nSENSITIVITY -- what happens when someone attacks an input")
    print(f"  {'change':<44}{'mean':>8}{'P(loss)':>10}{'worst 5%':>11}")
    cases = [
        ("as in the memo",                        {}),
        ("P/E fitted to my own scenarios",        dict(pe_intercept=14.05, pe_slope=0.243)),
        ("harsher P/E rule (10 + 0.30g)",         dict(pe_intercept=10.0, pe_slope=0.30)),
        ("bad regime 30% likely",                 dict(p_bear=0.30)),
        ("bad regime 40% likely",                 dict(p_bear=0.40)),
        ("bad regime 50% likely",                 dict(p_bear=0.50)),
        ("revenue spread 15% not 10%",            dict(base_rev_sd=0.15)),
        ("bad-case revenue $5.8B not $6.4B",      dict(bear_rev_mean=5800.0)),
        ("net margin 10.0% not 10.8%",            dict(margin_mean=0.100)),
        ("no net cash credited",                  dict(net_cash=0.0)),
        ("everything hostile at once",            dict(p_bear=0.40, pe_intercept=14.05,
                                                       pe_slope=0.243, margin_mean=0.102,
                                                       net_cash=0.0)),
    ]
    for label, kwargs in cases:
        r = simulate(**kwargs)
        p5 = np.percentile(r, 5)
        print(f"  {label:<44}{r.mean():>+8.1%}{(r < 0).mean():>10.1%}"
              f"{r[r <= p5].mean():>+11.1%}")


def plot(returns, path="mc_returns"):
    """One-column figure for the memo. Deliberately spare: the reader needs the
    shape of the distribution and two numbers, not bin densities, so there is
    no y-axis, no gridlines and no legend."""
    INK, SECOND, MUTED = "#0b0b0b", "#52514e", "#898781"
    AXIS = "#c3c2b7"
    GAIN, LOSS = "#c2c7cc", "#d03b3b"     # neutral fill vs status-critical

    p5 = np.percentile(returns, 5)
    cvar = returns[returns <= p5].mean()
    p_loss = (returns < 0).mean()
    median = np.median(returns)

    # Fine bins so the outline reads as a smooth shape rather than chunky bars.
    edges = np.arange(-0.76, 1.68, 0.04)
    counts, edges = np.histogram(returns, bins=edges)
    share = counts / len(returns) * 100
    centres = (edges[:-1] + edges[1:]) / 2
    top = share.max()

    fig, ax = plt.subplots(figsize=(3.6, 1.95))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Two filled regions split at zero, drawn as one continuous outline.
    ax.fill_between(centres, share, step="mid", where=centres < 0,
                    color=LOSS, linewidth=0, zorder=2)
    ax.fill_between(centres, share, step="mid", where=centres >= 0,
                    color=GAIN, linewidth=0, zorder=2)

    # The only reference line: today's price.
    ax.plot([0, 0], [0, top * 1.07], color=INK, lw=0.9, zorder=4)
    ax.text(0, top * 1.10, "today", fontsize=6, color=SECOND,
            ha="center", va="bottom", zorder=4)

    # Labels sit inside their own region, so nothing needs a leader line.
    ax.text(-0.33, top * 0.34, f"{p_loss:.0%}\nlose money", fontsize=7.2,
            color="white", ha="center", va="center", weight="bold",
            linespacing=1.25, zorder=5)
    ax.text(median, top * 0.30, f"median\n{median:+.0%}", fontsize=7.2,
            color=INK, ha="center", va="center", linespacing=1.25, zorder=5)
    ax.plot([median, median], [0, top * 0.14], color="white", lw=0.8, zorder=5)

    ax.set_title("Simulated 12-month return, 200,000 paths",
                 fontsize=7.5, color=INK, loc="left", pad=15, weight="bold")
    ax.text(0, 1.035, f"Worst 5% of paths average {cvar:.0%}".replace("-", "\u2212"),
            transform=ax.transAxes, fontsize=6.5, color=SECOND)

    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_xticks([-0.5, 0, 0.5, 1.0, 1.5])
    ax.set_xlim(-0.82, 1.66)
    ax.set_ylim(0, top * 1.24)
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=6, colors=MUTED, length=0, pad=3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.spines["bottom"].set_position(("outward", 2))

    fig.tight_layout(pad=0.25)
    fig.savefig(f"{path}.pdf", facecolor="white")      # vector, for LaTeX
    fig.savefig(f"{path}.png", dpi=240, facecolor="white")
    print(f"\nwrote {path}.pdf and {path}.png")


if __name__ == "__main__":
    returns = simulate()
    summarise(returns)
    sensitivity()
    plot(returns)
