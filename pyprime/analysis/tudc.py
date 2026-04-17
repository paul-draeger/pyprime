from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, List

from dataclasses import dataclass
from typing import ClassVar, Dict, List


from dataclasses import dataclass
from typing import ClassVar, Dict, List


class tudc:
    PALETTES = {
        "dark1": ["#2F57B2", "#7369BE", "#BC1589", "#D20F41", "#C85000"],
        "light1": ["#97C6FF", "#C8C8FF", "#FFB9FF", "#FFAAA5", "#FFBE78"],
        "dark2": ["#FFC700", "#767A23", "#007D4B", "#0A777F"],
        "light2": ["#FFE483", "#D2DC46", "#8CE6AA", "#8CE6D7"],

        "dark1_r": ["#C85000", "#D20F41", "#BC1589", "#7369BE", "#2F57B2"],
        "light1_r": ["#FFBE78", "#FFAAA5", "#FFB9FF", "#C8C8FF", "#97C6FF"],
        "dark2_r": ["#0A777F", "#007D4B", "#767A23", "#FFC700"],
        "light2_r": ["#8CE6D7", "#8CE6AA", "#D2DC46", "#FFE483"],

        "greys": [
            "#000000",
            "#323F4B",
            "#566371",
            "#7D8894",
            "#A5AEB8",
            "#D0D5DC",
            "#E7E9ED",
            "#FFFFFF",
        ],
    }

    def c(self, family, idx):
        family = family.lower()
        if family not in self.PALETTES:
            valid = ", ".join(sorted(self.PALETTES))
            raise ValueError(f"Unknown family '{family}'. Choose one of: {valid}")
        palette = self.PALETTES[family]
        return palette[idx % len(palette)]

    def palette(self, family):
        family = family.lower()
        if family not in self.PALETTES:
            valid = ", ".join(sorted(self.PALETTES))
            raise ValueError(f"Unknown family '{family}'. Choose one of: {valid}")
        return self.PALETTES[family].copy()





from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np



import numpy as np

def draw_zoom_box(ax, x0, x1, y0, y1, lw=0.8, color="gray", gap_pt=6):
    fig = ax.figure

    # points → data
    px = gap_pt * fig.dpi / 72.0
    inv = ax.transData.inverted()
    dx, dy = inv.transform((px, px)) - inv.transform((0, 0))

    # begrenzen
    dx = min(dx, 0.45 * abs(x1 - x0))
    dy = min(dy, 0.45 * abs(y1 - y0))

    z = -1000

    # Linien kompakt
    lines = [
        ([x0 + dx, x1 - dx], [y0, y0]),  # unten
        ([x0 + dx, x1 - dx], [y1, y1]),  # oben
        ([x0, x0], [y0 + dy, y1 - dy]),  # links
        ([x1, x1], [y0 + dy, y1 - dy]),  # rechts
    ]

    for xs, ys in lines:
        ax.plot(xs, ys, color=color, lw=lw, zorder=z)


def add_zoom_window(
    ax,
    zx,
    zy,
    wx="35%",
    wy="35%",
    conn=(0, 0, 1, 1),
    loc="upper right",
    pad=0.8,
    x=None,
    y=None,
    plot_func=None,
    rect_kwargs=None,
    conn_kwargs=None,
):
    """
    Fügt in eine bestehende Achse `ax` ein Zoom-Fenster ein.

    Parameter
    ----------
    ax : matplotlib.axes.Axes
        Hauptachse.
    zx : list | tuple
        x-Grenzen des Zoom-Bereichs, z. B. [xmin, xmax].
    zy : list | tuple
        y-Grenzen des Zoom-Bereichs, z. B. [ymin, ymax].
    wx : str | float
        Breite des Inset-Fensters, z. B. "35%" oder 2.0.
    wy : str | float
        Höhe des Inset-Fensters, z. B. "35%" oder 2.0.
    conn : list | tuple
        Vier Binärwerte für die Verbindungslinien:
        [unten-links, oben-links, unten-rechts, oben-rechts]
        1 = zeichnen, 0 = nicht zeichnen
    loc : str | int
        Position des Inset-Fensters in der Hauptachse.
    pad : float
        Abstand des Inset-Fensters vom Rand.
    x, y : array-like, optional
        Daten zum direkten Plotten im Inset, falls `plot_func` nicht gesetzt ist.
    plot_func : callable, optional
        Funktion mit Signatur `plot_func(axins)`, die den Inhalt des
        Inset-Fensters zeichnet.
    rect_kwargs : dict, optional
        kwargs für das Rechteck des Zoom-Bereichs.
    conn_kwargs : dict, optional
        kwargs für die Verbindungslinien.

    Returns
    -------
    axins : matplotlib.axes.Axes
        Die Inset-Achse.
    """
    zx = np.asarray(zx, dtype=float)
    zy = np.asarray(zy, dtype=float)

    if zx.shape != (2,) or zy.shape != (2,):
        raise ValueError("zx und zy müssen jeweils genau zwei Werte enthalten.")

    if len(conn) != 4:
        raise ValueError("conn muss vier Einträge haben: [bl, tl, br, tr].")

    rect_kwargs = {} if rect_kwargs is None else rect_kwargs.copy()
    conn_kwargs = {} if conn_kwargs is None else conn_kwargs.copy()

    rect_defaults = dict(fill=False, lw=1.2)
    conn_defaults = dict(color="black", lw=1.0)
    rect_defaults.update(rect_kwargs)
    conn_defaults.update(conn_kwargs)

    x0, x1 = zx
    y0, y1 = zy

    draw_zoom_box(
        ax,
        x0, x1, y0, y1,
        lw=0.8,
        color="k",
        gap_pt=1
    )

    axins = inset_axes(ax, width=wx, height=wy, loc=loc, borderpad=pad)

    if plot_func is not None:
        plot_func(axins)
    elif x is not None and y is not None:
        axins.plot(x, y)
    else:
        for line in ax.lines:
            axins.plot(
                line.get_xdata(),
                line.get_ydata(),
                color=line.get_color(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
                marker=line.get_marker(),
                markersize=line.get_markersize(),
                zorder=line.get_zorder(),
            )

    axins.set_xlim(x0, x1)
    axins.set_ylim(y0, y1)

    corners_main = [
        (x0, y0),  # 0: unten-links
        (x0, y1),  # 1: oben-links
        (x1, y0),  # 2: unten-rechts
        (x1, y1),  # 3: oben-rechts
    ]

    corners_inset = [
        (0, 0),  # 0
        (0, 1),  # 1
        (1, 0),  # 2
        (1, 1),  # 3
    ]

    for j, i in enumerate(conn):
        if i >= 0:
            xm, ym = corners_main[j]
            xi, yi = corners_inset[i]

            ax.annotate(
                "",
                xy=(xm, ym),
                xycoords=ax.transData,
                xytext=(xi, yi),
                textcoords=axins.transAxes,
                arrowprops=dict(
                    arrowstyle="-",
                    color="k",
                    lw=0.8,
                    zorder=-1000,
                ),
            )

    return axins
