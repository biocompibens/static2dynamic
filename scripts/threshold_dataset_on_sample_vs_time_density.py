import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main(args: Namespace):
    # Checks
    assert Path(args.save_base_path).exists(), f"Save path {args.save_base_path} does not exist"

    # Loads
    time_preds_path = Path(args.time_preds_path)
    print(f"\nLoading time predictions from:\n{time_preds_path}... ", end=" ", flush=True)
    time_preds = pd.read_parquet(time_preds_path)
    print("done")
    tot_nb_samples = len(time_preds)
    print(f"\nTotal number of samples: {tot_nb_samples}")
    print(f"Number of bins: {args.nbins}")
    uniform_density = tot_nb_samples / args.nbins
    print(f"Theoretical number of samples per bin were the distribution uniform: {uniform_density:.1f}")
    print(f"Passed manual threshold: {args.threshold}")

    labels = np.array(time_preds["true_label"].values)
    times = np.array(time_preds["time"].values)

    # Figure
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(18, 12))

    try:
        sorted_uniq_labels = np.unique(labels.astype(np.uint8), sorted=True)
    except ValueError as e:
        try:
            uniq_labels = np.unique(labels)
            sorted_uniq_labels = np.array(sorted(uniq_labels, key=lambda x: int(x.split("_")[-1])))
        except ValueError as e2:
            print(f"\nCould not sort labels as integers: {e}\nand: {e2}.\nUsing original labels instead")
            sorted_uniq_labels = np.unique(labels)
    sorted_uniq_labels = [str(lab) for lab in sorted_uniq_labels]

    # Top: stacked histogram
    sns.histplot(
        x=times,
        hue=labels,
        bins=args.nbins,
        palette="viridis",
        multiple="stack",
        ax=ax1,
        hue_order=sorted_uniq_labels,
        legend=False,
    )
    uniform_line = ax1.axhline(
        uniform_density, color="black", linestyle="dotted", label=f"Theoretical uniform density ({uniform_density:.1f})"
    )
    thresh_line = ax1.axhline(args.threshold, color="red", linestyle="--", label=f"Threshold ({args.threshold})")
    palette = sns.color_palette("viridis", n_colors=len(sorted_uniq_labels))
    patches = [
        mpatches.Patch(color=palette[i], label=str(sorted_uniq_labels[i])) for i in range(len(sorted_uniq_labels))
    ]
    ax1.set_title("Stacked histogram of original continuous time predictions")
    ax1.set_xlabel("Continuous time prediction")
    ax1.set_ylabel("Raw count")
    ax1.legend(
        handles=patches + [uniform_line, thresh_line],
        labels=[p.get_label() for p in patches] + [uniform_line.get_label(), thresh_line.get_label()],
    )
    # Middle: same histogram as above but with args.threshold * 2 as y limit
    sns.histplot(
        x=times,
        hue=labels,
        bins=args.nbins,
        palette="viridis",
        multiple="stack",
        ax=ax2,
        hue_order=sorted_uniq_labels,
        legend=False,
    )
    ax2.set_title("Zoomed stacked histogram of original continuous time predictions")
    ax2.set_xlabel("Continuous time prediction")
    ax2.set_ylabel("Raw count")
    ax2.set_ylim(0, args.threshold * 2)
    ax2.axhline(
        uniform_density, color="black", linestyle="dotted", label=f"Theoretical uniform density ({uniform_density:.1f})"
    )
    ax2.axhline(args.threshold, color="red", linestyle="--", label=f"threshold ({args.threshold})")

    # Bottom: show line of time with darker gray at times above args.threshold
    counts, bins_borders = np.histogram(times, bins=args.nbins)
    assert len(counts) == args.nbins, "Counts length must match number of bins"
    assert len(bins_borders) == args.nbins + 1, "Bins borders length must be number of bins + 1"
    # fill whome timeline with light gray
    ymin, ymax = -0.2, 0.2
    ax3.fill_between([0, args.nbins], ymin, ymax, color="lightgray")
    # draw dark gray patches where counts are above threshold
    counts_above_thresh = counts > args.threshold
    list_of_time_bins_above_thresh = []  # [(0.1, 0.3), (0.5, 0.57), ...]
    start = None
    for i, val in enumerate(counts_above_thresh):
        if val and start is None:
            start = i
        elif not val and start is not None:
            ax3.fill_between([start, i], ymin, ymax, color="dimgray")
            list_of_time_bins_above_thresh.append((bins_borders[start], bins_borders[i]))
            start = None
    # edge case: end of array is True
    if start is not None:
        ax3.fill_between([start, args.nbins], ymin, ymax, color="dimgray")
        list_of_time_bins_above_thresh.append((bins_borders[start], bins_borders[args.nbins]))
    # annotate transitions with true time values
    for start, end in list_of_time_bins_above_thresh:
        ax3.text(
            np.where(bins_borders == start)[0],
            ymin - 0.1,
            f"{start:.2f}".lstrip("0"),
            ha="center",
            va="bottom",
            fontsize=8,
        )
        ax3.text(
            np.where(bins_borders == end)[0], ymin - 0.1, f"{end:.2f}".lstrip("0"), ha="center", va="bottom", fontsize=8
        )
    # make nice
    ax3.set_ylim(-1, 1)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.spines["bottom"].set_visible(False)
    ax3.spines["left"].set_visible(False)
    ax3.set_title(
        f"Times above threshold: {np.sum(counts_above_thresh)} / {args.nbins} ({np.round(np.sum(counts_above_thresh) / args.nbins * 100, 2)}%) for {args.nbins} bins",
        y=0.7,
    )

    # Misc. fig
    fig.suptitle(
        f"Thresholding continuous time predictions at threshold={args.threshold} on dataset '{time_preds_path.name.split('__')[0]}' with {args.nbins} bins",
        fontsize=16,
    )
    fig.text(0.5, 0.94, f"(at {time_preds_path})", ha="center", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    # Save fig
    fig_save_path = Path(args.save_base_path) / f"thresholded_times_threshold={args.threshold}_nbins={args.nbins}.png"
    fig.savefig(fig_save_path, dpi=300)
    print(f"\nSaved figure to:\n{fig_save_path}")

    # Save times above threshold
    list_of_time_bins_above_thresh_savepath = (
        Path(args.save_base_path) / f"times_above_threshold={args.threshold}_nbins={args.nbins}.pickle"
    )
    with open(list_of_time_bins_above_thresh_savepath, "wb") as f:
        pickle.dump(list_of_time_bins_above_thresh, f)
    print(
        f"\nSaved {len(list_of_time_bins_above_thresh)} time bins above threshold to {list_of_time_bins_above_thresh_savepath}:"
    )
    print([(round(float(start), 2), round(float(end), 2)) for start, end in list_of_time_bins_above_thresh])


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Threshold dataset w.r.t. the density of samples along time. Outputs a figure and a list of time bins above threshold."
    )
    parser.add_argument(
        "--time_preds_path",
        type=str,
        required=True,
        help="Path to the time predictions parquet file",
    )
    parser.add_argument(
        "--save_base_path",
        type=str,
        required=True,
        help="Path to save the resulting figure and args.thresholded times/samples dataframe",
    )
    parser.add_argument(
        "--nbins",
        type=int,
        required=True,
        help="Number of bins for the histogram",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        required=True,
        help="Threshold value to apply on the continuous time predictions (in raw counts); beware: this is dependent on the number of bins!",
    )
    args = parser.parse_args()
    main(args)
