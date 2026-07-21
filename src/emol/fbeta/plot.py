import numpy as np
from .fbeta import *

from itertools import zip_longest

from scipy.stats import mannwhitneyu

LEGEND_SIZE = 8

def _ax_config(ax, betas):
    ax.grid(ls=":")
    ax.set_xscale("log")
    ax.set_xlim(betas[0], betas[-1])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("$F_\\beta$ score")
    # ax.set_xlabel("$\\beta$")
    ax.spines[["right", "top"]].set_visible(False)

def fbeta_plot(series, labels, ax=None, lim=DEFAULT_LIM, N=DEFAULT_N, add_legend=True):
    if ax is None:
        ax = plt.gca()

    betas = np.geomspace(1 / lim, lim, N)
    _ax_config(ax, betas)

    f_betas = fbeta(series[:, 0], series[:, 1], beta=betas)
    best = np.argmax(f_betas, axis=0)
    _, c_i = np.unique(best, return_index=True)
    candidates = best[np.sort(c_i)]

    for fb in f_betas:
        ax.plot(betas, fb, c="#CCCCCC", zorder=1)

    cross_mat = fbeta_crossing_mat(series)
    changing_points = []

    for c, c_next in zip_longest(candidates, candidates[1:]):
        plot_map = best == c
        fix = plot_map[-1]
        plot_map[np.logical_xor(plot_map, np.roll(plot_map, -1))] = True
        plot_map[-1] = fix

        if c_next is not None:
            changing_points.append(cross_mat[c, c_next])

        ax.plot(betas[plot_map], f_betas[c][plot_map], label=labels[c])

    for a in np.array(changing_points):
        ax.vlines(a, 0.0, 1.00, color="k", lw=1.2, ls=":")

        ax.text(
            a,
            1.0,
            f"{a:.2f}",
            rotation=270,
            fontsize="x-small",
            verticalalignment="top",
            horizontalalignment="left",
        )

    if add_legend:
        ax.legend(
            bbox_to_anchor=(0.5, 1.2, 0, 0),
            loc="upper center",
            ncol=4,
            prop=dict(family="monospace", size=LEGEND_SIZE),
        )


def fbeta_plot_cv(series, labels, ax=None, lim=DEFAULT_LIM, N=DEFAULT_N):
    if ax is None:
        ax = plt.gca()

    betas = np.geomspace(1 / lim, lim, N)
    _ax_config(ax, betas)

    f_betas = fbeta(series[..., 0], series[..., 1], beta=betas)

    f_betas_mean = f_betas.mean(axis=0)
    f_betas_std = f_betas.std(axis=0)

    best = np.argmax(f_betas_mean, axis=0)
    _, c_i = np.unique(best, return_index=True)
    candidates = best[np.sort(c_i)]

    for fb, fbs in zip(f_betas_mean, f_betas_std):
        ax.plot(betas, fb, c="#CCCCCC", zorder=1)
        ax.fill_between(betas, fb + fbs, fb - fbs, alpha=0.01, color="#CCCCCC")

    changing_points = []

    stats = np.zeros_like(betas, dtype=bool)

    for c, c_next in zip_longest(candidates, candidates[1:]):
        plot_map = best == c
        fix = plot_map[0]
        plot_map[np.logical_xor(plot_map, np.roll(plot_map, 1))] = True
        plot_map[0] = fix

        if c_next is not None:
            changing_points.append(
                crossing_approximations(
                    f_betas_mean[c], f_betas_mean[c_next], space=betas
                )[0]
            )

        bts = betas[plot_map]
        fbm = f_betas_mean[c][plot_map]
        fbs = f_betas_std[c][plot_map]

        pl = ax.plot(bts, fbm, label=labels[c], zorder=8)
        color = pl[0].get_color()

        ax.fill_between(bts, fbm + fbs, fbm - fbs, alpha=0.3, color=color, zorder=7)
        ax.plot(bts, fbm + fbs, ls=":", lw=0.5, color=color)
        ax.plot(bts, fbm - fbs, ls=":", lw=0.5, color=color)

        for i in np.argwhere(plot_map):
            i_stats = []
            for j in range(series.shape[1]):
                if j == c:
                    continue

                s, p = mannwhitneyu(f_betas[:, c, i], f_betas[:, j, i])
                i_stats.append(p < 0.05)

            stats[i] = np.all(i_stats)

    ax.scatter(betas[stats], np.repeat(0.01, np.sum(stats)), s=40, marker="s", c="k")

    for a in np.array(changing_points):
        ax.vlines(a, 0.0, 1.00, color="k", lw=1.2, ls=":", zorder=9)

        ax.text(
            a,
            1.0,
            f"{a:.2f}",
            rotation=270,
            fontsize="x-small",
            verticalalignment="top",
            horizontalalignment="left",
        )

    ax.legend(
        bbox_to_anchor=(0.5, 1.2, 0, 0),
        loc="upper center",
        ncol=4,
        prop=dict(family="monospace", size=LEGEND_SIZE),
    )


def fbeta_plot_with_ref(
    series, ref, labels, ref_label, ax=None, lim=DEFAULT_LIM, N=DEFAULT_N
):
    if ax is None:
        ax = plt.gca()

    betas = np.geomspace(1 / lim, lim, N)
    _ax_config(ax, betas)

    f_betas = fbeta(series[:, 0], series[:, 1], beta=betas)
    best = np.argmax(f_betas, axis=0)
    _, c_i = np.unique(best, return_index=True)
    candidates = best[np.sort(c_i)]

    ref_fbeta = combined_curve(fbeta(ref[:, 0], ref[:, 1], beta=betas))
    ax.plot(betas, ref_fbeta, c="k", ls="--", lw=1.5, label=ref_label, zorder=7)

    for fb in f_betas:
        ax.plot(betas, fb, c="#CCCCCC", zorder=1)

    ref_better = combined_curve(f_betas) > ref_fbeta

    changing_points = []

    for c, c_next in zip_longest(candidates, candidates[1:]):
        plot_map = np.logical_and(ref_better, best == c)
        fix = plot_map[-1]
        plot_map[np.logical_xor(plot_map, np.roll(plot_map, -1))] = True
        plot_map[-1] = fix

        if np.all(~plot_map):
            continue

        fix = plot_map[0]
        plot_map[np.logical_xor(plot_map, np.roll(plot_map, 1))] = True
        plot_map[0] = fix

        if c_next is not None:
            changing_points.extend(
                crossing_approximations(
                    f_betas[c][plot_map],
                    f_betas[c_next][plot_map],
                    space=betas[plot_map],
                )
            )

        changing_points.extend(
            crossing_approximations(
                f_betas[c][plot_map], ref_fbeta[plot_map], space=betas[plot_map]
            )
        )

        ax.plot(betas[plot_map], f_betas[c][plot_map], label=labels[c])

    for a in np.array(changing_points):
        ax.vlines(a, 0.0, 1.00, color="k", lw=1.2, ls=":")

        ax.text(
            a,
            1.0,
            f"{a:.2f}",
            rotation=270,
            fontsize="x-small",
            verticalalignment="top",
            horizontalalignment="left",
        )

    ax.legend(
        bbox_to_anchor=(0.5, 1.2, 0, 0),
        loc="upper center",
        ncol=4,
        prop=dict(family="monospace", size=LEGEND_SIZE),
    )



def fbeta_plot_combined_cv(series, labels, ax=None, lim=DEFAULT_LIM, N=DEFAULT_N):
    if ax is None:
        ax = plt.gca()

    betas = np.geomspace(1 / lim, lim, N)
    _ax_config(ax, betas)

    split_betas = []

    for split_data in series:
        f_betas = []

        for moo_results in split_data:
            fb = fbeta(moo_results[0, :], moo_results[1, :], beta=betas)
            f_betas.append(combined_curve(fb))

        split_betas.append(f_betas)

    split_betas = np.array(split_betas)

    f_betas_mean = []

    stats = []

    for i in split_betas.T:
        s, p = mannwhitneyu(i[0], i[1])
        stats.append(p < 0.05)

    stats = np.array(stats)

    ax.scatter(betas[stats], np.repeat(0.01, np.sum(stats)), s=10, marker="s", c="k")

    for c_i, curves in enumerate(np.rollaxis(split_betas, 1)):
        fbm = np.mean(curves, axis=0)
        fbs = np.std(curves, axis=0)

        pl = ax.plot(betas, fbm, lw=1.5, ls="--", label=labels[c_i])
        color = pl[0].get_color()
        ax.plot(betas, curves.T, alpha=0.1, c=color, lw=1)
        ax.fill_between(betas, fbm + fbs, fbm - fbs, alpha=0.3, color=color)
        ax.plot(betas, fbm + fbs, ls=":", lw=0.5, color=color)
        ax.plot(betas, fbm - fbs, ls=":", lw=0.5, color=color)

        f_betas_mean.append(fbm)

    # TODO: add for more cases
    changing_points = crossing_approximations(
        f_betas_mean[0], f_betas_mean[1], space=betas
    )

    for a in np.array(changing_points):
        ax.vlines(a, 0.0, 1.00, color="k", lw=1.2, ls=":")

        ax.text(
            a,
            1.0,
            f"{a:.2f}",
            rotation=270,
            fontsize="x-small",
            verticalalignment="top",
            horizontalalignment="left",
        )

    ax.legend(
        bbox_to_anchor=(0.5, 1.2, 0, 0),
        loc="upper center",
        ncol=4,
        prop=dict(family="monospace", size=LEGEND_SIZE),
    )
