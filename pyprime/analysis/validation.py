from .tudc import tudc
from .tudc import add_zoom_window

import numpy as np


def save_axis(ax, filename):
    import matplotlib.pyplot as plt

    fig2, ax2 = plt.subplots(figsize=ax.figure.get_size_inches())

    # Linien kopieren
    for line in ax.lines:
        ax2.plot(
            line.get_xdata(),
            line.get_ydata(),
            color=line.get_color(),
            linestyle=line.get_linestyle(),
            linewidth=line.get_linewidth(),
            marker=line.get_marker(),
            markersize=line.get_markersize(),
        )

    # Limits übernehmen
    ax2.set_xlim(ax.get_xlim())
    ax2.set_ylim(ax.get_ylim())

    # Labels + Titel
    ax2.set_xlabel(ax.get_xlabel())
    ax2.set_ylabel(ax.get_ylabel())
    ax2.set_title(ax.get_title())

    # Legend (falls vorhanden)
    if ax.get_legend() is not None:
        handles, labels = ax.get_legend_handles_labels()
        ax2.legend(handles, labels)

    fig2.tight_layout()
    fig2.savefig(filename)
    plt.close(fig2)


def local_poly_filter(t, x, w=2, deg=2):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)

    y = np.empty_like(x)
    half = 0.5 * w

    for i, ti in enumerate(t):
        mask = (t >= ti - half) & (t <= ti + half)

        tj = t[mask]
        xj = x[mask]

        if len(tj) < deg + 1:
            y[i] = x[i]
            continue

        p = np.polyfit(tj - ti, xj, deg)
        y[i] = np.polyval(p, 0.0)

    return y


def plot_case_ascent(filename, sa, outname, outname_sep, case_label="Case A", save_timestamp=True,t_e=None, averaging_window=0,zx=[0,0.35],zy=[1,2.1]):
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import rc

    from pyprime.analysis.sim import sim
    from pyprime.analysis.tudc import tudc, add_zoom_window

    rc('text', usetex=True)
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Noto Sans']})
    plt.rcParams.update({'font.size': 10})

    tud1 = tudc()

    lstyl = ['--', '-.', (0, (1, 1))]
    lwi = 1.5
    bascol = tud1.c('greys', 4)
    c_pal = 'dark1'
    labels = [r'$D / h = 48$', r'$D / h = 24$', r'$D / h = 12$']

    fig, ax = plt.subplots(1, 2, figsize=(6, 2.5))
    fig.subplots_adjust(wspace=0.25, hspace=0.25)

    fig1, ax1 = plt.subplots(1, 1, figsize=(2.5, 2.5))
    fig2, ax2 = plt.subplots(1, 1, figsize=(2.5, 2.5))


    with open(filename, 'r') as f:
        lines = f.readlines()

    data_lines = [line for line in lines if line.strip() and line[0].isdigit()]
    data = np.array([[float(num) for num in line.split()] for line in data_lines])

    time_b = data[:, 0]
    velocity_z = data[:, 5]
    if (averaging_window>0):
        velocity_z = local_poly_filter(time_b,velocity_z,w=averaging_window)

    ax[0].plot(
        time_b, velocity_z,
        label='Basilisk',
        color=bascol,
        zorder=1,
        linestyle='-',
        lw=lwi
    )
    ax1.plot(
        time_b, velocity_z,
        label='Basilisk',
        color=bascol,
        zorder=1,
        linestyle='-',
        lw=lwi
    )

    dvdt = np.gradient(velocity_z, time_b)
    ax[1].plot(
        time_b, dvdt,
        label='Basilisk',
        color=bascol,
        zorder=1,
        linestyle='-',
        lw=lwi
    )
    ax2.plot(
        time_b, dvdt,
        label='Basilisk',
        color=bascol,
        zorder=1,
        linestyle='-',
        lw=lwi
    )


    for k, si in enumerate(sa):
        b1 = si.b(1)
        coli = tud1.c(c_pal, k * 2)
        bub_w = b1.w
        if (averaging_window>0):
            bub_w = local_poly_filter(b1.t,bub_w,w=averaging_window)

        ax[0].plot(
            b1.t, bub_w,
            linestyle=lstyl[k],
            color=coli,
            label=labels[k],
            zorder=10 - k,
            lw=lwi
        )
        ax1.plot(
            b1.t, bub_w,
            linestyle=lstyl[k],
            color=coli,
            label=labels[k],
            zorder=10 - k,
            lw=lwi
        )

        dwdt = np.gradient(bub_w, b1.t)
        t_dwdt = b1.t

        ax[1].plot(
            t_dwdt[2:], dwdt[2:],
            linestyle=lstyl[k],
            color=coli,
            label=labels[k],
            zorder=10 - k,
            lw=lwi
        )
        ax2.plot(
            t_dwdt[2:], dwdt[2:],
            linestyle=lstyl[k],
            color=coli,
            label=labels[k],
            zorder=10 - k,
            lw=lwi
        )

    if t_e is None:
        t_tresh = 0.99
        umax = np.max(velocity_z)
        i_end = np.argmin(np.abs(velocity_z - umax * t_tresh))
        t_e = time_b[i_end]
    else:
        i_end = np.argmin(np.abs(t_e - time_b))

    u_max_plot = np.max(velocity_z[:i_end])
    u_min_plot = np.min(velocity_z[:i_end])

    du_max_plot = np.max(dvdt[:i_end])
    du_min_plot = np.min(dvdt[:i_end])
    plot_margin_fac = 1.1
    pmar = (plot_margin_fac-1)*u_max_plot
    dpmar = (plot_margin_fac-1)*du_max_plot

    ax[0].set_xlim(-0.1, t_e)
    ax[0].set_ylim(0, u_max_plot+pmar)

    ax[0].set_xlabel(r'$t \, \sqrt{g / d}$')
    ax[0].set_ylabel(r'$u / \sqrt{g \, d}$')
    ax[0].legend(frameon=False)
    ax[0].set_title(f'Ascent velocity {case_label}', fontsize=10)

    ax1.set_xlim(-0.1, t_e)
    ax1.set_ylim(0, u_max_plot+pmar)

    ax1.set_xlabel(r'$t \, \sqrt{g / d}$')
    ax1.set_ylabel(r'$u / \sqrt{g \, d}$')
    ax1.legend(frameon=False)
    ax1.set_title(f'Ascent velocity {case_label}', fontsize=10)

    ax[1].set_xlim(-0.1, t_e)
    ax[1].set_ylim(du_min_plot-dpmar, du_max_plot+dpmar)
    ax[1].set_xlabel(r'$t \, \sqrt{g / d}$')
    ax[1].set_ylabel(r'$\dot{u} / g $')
    ax[1].set_title(f'Acceleration {case_label}', fontsize=10)

    ax[1] = add_zoom_window(
        ax[1],
        zx=zx,
        zy=zy,
        wx=1,
        wy=1,
        loc='upper right',
        conn=[-1, -1, 0, 1]
    )

    ax2.set_xlim(-0.1, t_e)
    ax2.set_ylim(du_min_plot-dpmar, du_max_plot+dpmar)
    ax2.set_xlabel(r'$t \, \sqrt{g / d}$')
    ax2.set_ylabel(r'$\dot{u} / g $')
    ax2.set_title(f'Acceleration {case_label}', fontsize=10)

    ax2 = add_zoom_window(
        ax2,
        zx=zx,
        zy=zy,
        wx=1,
        wy=1,
        loc='upper right',
        conn=[-1, -1, 0, 1]
    )

    fig.savefig(outname, bbox_inches='tight', pad_inches=0.1)

    if save_timestamp:
        stem, ext = outname.rsplit('.', 1)
        ts = int(time.time())
        fig.savefig(f"{stem}_{ts}.{ext}", bbox_inches='tight', pad_inches=0.1)


    # --- neue Subfigures ---
    stem, ext = outname_sep.rsplit('.', 1)

    fig1.savefig(f"{stem}_fig1.{ext}", bbox_inches='tight', pad_inches=0.1)
    fig2.savefig(f"{stem}_fig2.{ext}", bbox_inches='tight', pad_inches=0.1)

    #return fig, ax

from .tudc import tudc, add_zoom_window
import numpy as np


def plot_case_timestep(
    filename,
    sa,
    outname,
    case_label="Case A",
    outname_sep=None,
    save_timestamp=True,
    t_e=None,
    zx=None,
    zy=None,
):
    import time
    import matplotlib.pyplot as plt
    from matplotlib import rc

    rc('text', usetex=True)
    rc('font', **{'family': 'sans-serif', 'sans-serif': ['Noto Sans']})
    plt.rcParams.update({'font.size': 10})

    tud1 = tudc()

    lstyl = ['--', '-.', (0, (1, 1))]
    lwi = 1.5
    bascol = tud1.c('greys', 4)
    c_pal = 'dark1'
    labels = [r'$D / h = 48$', r'$D / h = 24$', r'$D / h = 12$']

    fig, ax = plt.subplots(figsize=(3, 2.5))

    # Basilisk / VOF data
    with open(filename, 'r') as f:
        lines = f.readlines()

    data_lines = [line for line in lines if line.strip() and line[0].isdigit()]
    data = np.array([[float(num) for num in line.split()] for line in data_lines])

    time_b = data[:, 0]
    dt_b = np.diff(time_b)/10
    tmid_b = 0.5 * (time_b[:-1] + time_b[1:])

    ax.plot(
        tmid_b,
        dt_b,
        label='Basilisk',
        color=bascol,
        zorder=1,
        linestyle='-',
        lw=lwi,
    )
    print('\n\n### DT--{case_label} ###')
    # PRIME data
    for k, si in enumerate(sa):
        b1 = si.b(1)

        dt = np.diff(b1.t)
        tmid = 0.5 * (b1.t[:-1] + b1.t[1:])

        coli = tud1.c(c_pal, k * 2)

        ax.plot(
            tmid,
            dt,
            linestyle=lstyl[k],
            color=coli,
            label=labels[k],
            zorder=10 - k,
            lw=lwi,
        )
        indco = np.argmin(np.abs(b1.t-8))
        tco = b1.t[indco]-b1.t[indco-1]


        Ddx = 1/si.dx
        print(f'{Ddx:.1f} - dt(t=8) = {tco:.3e}')

    print('### END--{case_label} ###')

    ax.set_xlim(-0.1, t_e)
    #ax.set_ylim(max(0.0, dt_min_plot - pmar_lo), dt_max_plot + pmar_up)

    ax.set_xlabel(r'$t \, \sqrt{g / d}$')
    ax.set_ylabel(r'$\Delta t \, \sqrt{g / d}$')
    ax.set_title(f'Timestep {case_label}', fontsize=10)
    ax.legend(frameon=False)




    fig.savefig(outname)

    if outname_sep is not None:
        fig.savefig(outname_sep)

    if save_timestamp:
        stem, ext = outname.rsplit('.', 1)
        ts = int(time.time())
        fig.savefig(f"{stem}_{ts}.{ext}")

    return fig, ax
