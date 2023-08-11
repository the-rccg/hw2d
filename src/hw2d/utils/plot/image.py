import fire
import numpy as np
import h5py
from tqdm import tqdm
from typing import Dict, Union, List, Tuple
from matplotlib.colors import Colormap
import matplotlib.pyplot as plt

from hw2d.utils.plot.movie import get_extended_viridis
from hw2d.utils.latex_format import latex_format


def plot_dict(
    plot_dic: Dict,
    cmap: Union[str, Colormap, List[str]],
    couple_cbars: bool = False,
    figsize: Tuple or None = None,
    sharex: bool = False,
    sharey: bool = False,
    vertical: bool = False,
    equal: bool = True,
    cbar_label_spacing: float or None = None,
):
    n = len(plot_dic.values())
    labels = list(plot_dic.keys())
    if figsize is None:
        figsize = (n * 4, 4)
    if vertical:
        fig, axarr = plt.subplots(n, 1, figsize=figsize, sharex=sharex)
    else:
        fig, axarr = plt.subplots(
            1, n, figsize=figsize, sharey=sharey, constrained_layout=True
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
            imgs.append(ax.imshow(data, cmap=cmap))
            cbars.append(plt.colorbar(imgs[-1], pad=0, fraction=0.05))
            cax = cbars[-1].ax
            if cbar_label_spacing is not None:
                cax.yaxis.set_tick_params(pad=cbar_label_spacing)
            cax.xaxis.set_tick_params(pad=0)
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
    file_path: str = "_512.h5",
    t: float = 200,
    save_type: str = "jpg",
    zero_omega: bool = True,
    # save_path: str = "imgs/raw/gridsize=512x512_k0=0.15_c1=1.0_dt=0.025_arak=1_kappa=1_N=3_nu=5.0e-06_scale=0.01_age=1000.000_version=6.pdf",
    ticks: bool = False,
):
    plot_order = ["density", "omega", "phi"]
    save_file_path = file_path.replace(".h5", ".jpg") + f"_t={t}"

    # Setting up paths and stuff
    if isinstance(t, float) and t.is_integer():
        t = int(t)

    file = h5py.File(file_path, "r")
    parameters = dict(file.attrs)

    t_idx = int(t // parameters["frame_dt"])
    print(t_idx)

    # Structure data
    plot_dic = {field_name: file[field_name][t_idx] for field_name in plot_order}
    x, y = plot_dic[plot_order[0]].shape

    # Center Omega?
    if zero_omega:
        plot_dic["omega"] = plot_dic["omega"] - np.mean(plot_dic["omega"])

    # Name Properly
    plot_dic = {latex_format(name): value for name, value in plot_dic.items()}

    ve = get_extended_viridis(vals=600)

    # Plot
    fig = plot_dict(
        plot_dic,
        cmap=ve,
        couple_cbars=False,
        figsize=None,
        sharex=True,
        sharey=True,
        vertical=False,
        cbar_label_spacing=1.7,
    )

    for i, ax in enumerate(fig.get_axes()):
        if i >= len(plot_order):
            break
        if ticks:
            ax.set_xticks([0, x // 2, x])
            if i == 0:
                ax.set_yticks([0, y // 2, y])
        else:
            ax.set_xticks([])
            ax.set_yticks([])

    # Adjust engine again
    fig.get_layout_engine().set(w_pad=0.013, h_pad=0.01, hspace=0, wspace=0)

    if save_file_path:
        fig.savefig(save_file_path[:-4] + ".jpg")
        fig.savefig(save_file_path[:-4] + ".pdf")
    print(save_file_path)


if __name__ == "__main__":
    main()
