from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import latexplotlib as lpl
from tqdm import tqdm
import fire
import h5py


def plot_dict(
    plot_dic,
    cmap,
    couple_cbars=False,
    figsize=None,
    sharex=False,
    sharey=False,
    vertical=False,
    equal=True,
    cbar_label_spacing=None,
):
    n = len(plot_dic.values())
    labels = list(plot_dic.keys())
    if figsize is None:
        figsize = (n * 4, 4)
    if vertical:
        fig, axarr = lpl.subplots(n, 1, figsize=figsize, sharex=sharex)
    else:
        fig, axarr = lpl.subplots(
            1, n, figsize=figsize, ratio=1, constrained_layout=True
        )
    vmin = 0
    vmax = 0
    imgs = []
    cbars = []
    for i in tqdm(range(n)):
        # Local Names
        label = labels[i]
        data = plot_dic[label]
        ax = axarr[i]
        if len(data.shape) == 1:
            ax.axhline(0, color="k", alpha=0.7, linewidth=1)
            ax.plot(data)
        elif len(data.shape) == 2:
            imgs.append(ax.imshow(data, cmap=cmap))  # , interpolation='nearest'))
            cbars.append(plt.colorbar(imgs[-1], pad=0, fraction=0.05))  # , cax=cax
            cax = cbars[-1].ax
            if cbar_label_spacing is not None:
                cax.yaxis.set_tick_params(pad=cbar_label_spacing)
            cax.xaxis.set_tick_params(pad=0)
            # cbars.append(plt.colorbar(imgs[-1], cax=cax))
        else:
            raise BaseException(
                f"Shape not recognized.  {label} has shape {data.shape}"
            )
        # Properties
        ax.set_title(label, pad=4)
        vmin = min(np.min(data), vmin)
        vmax = max(np.max(data), vmax)
        if equal:
            ax.set_aspect("equal")
    # fix colorbars together
    if couple_cbars:
        for img in imgs:
            img.set_clim(vmin, vmax)
    return fig


def main(
    file_path: str = "test.h5",
    t: float = 300,
    field_list: List = ("density", "omega", "phi"),
):
    with h5py.File(file_path, "r") as hf:
        parameters = dict(hf.attrs)
        t_idx = t // parameters["dt"]
        field_data = {field_name: hf[field_name][t_idx] for field_name in field_list}
        fig = plot_dict(field_data, cmap=ve)
        fig.savefig(file_path.replace(".h5", ".jpg"))


if __name__ == "__main__":
    fire.Fire(main)
