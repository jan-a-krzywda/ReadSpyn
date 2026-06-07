import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


# plot coulomb peak and operating point

def plot_coulomb_peak(params_coulomb_peak, span=3.0, points=1000, ax=None):
    """
    Plot the Coulomb-peak conductance G(ε) and highlight the chosen operating point.

    Parameters
    ----------
    params_coulomb_peak : dict
        {
          'g0': <float>,        # peak conductance (S)
          'eps0': <float>,      # operating point relative to eps_width (unitless)
          'eps_width': <float>  # energy width (eV)
        }
        Model used: G(ε) = g0 / cosh^2( 2 * ε / eps_width )

    span : float, optional
        Plot ε in the range [-span * eps_width, +span * eps_width].
    points : int, optional
        Number of samples in the ε grid.
    ax : matplotlib.axes.Axes or None
        If provided, draw on this Axes; otherwise create a new one.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The Axes with the plot.
    """
    g0       = float(params_coulomb_peak['g0'])
    eps0_rel = float(params_coulomb_peak['eps0'])
    eps_w    = float(params_coulomb_peak['eps_width'])

    # ε grid and conductance profile (sech^2 shape)
    eps = np.linspace(-span * eps_w, span * eps_w, points)
    G   = g0 / np.cosh(2.0 * eps / eps_w)**2

    # Operating point (given relative to eps_width)
    eps_pick = eps0_rel * eps_w
    G_pick   = g0 / np.cosh(2.0 * eps_pick / eps_w)**2

    # Make / use axis
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4), dpi=100)

    ax.plot(eps, G, lw=2, label='Coulomb peak')
    ax.axvline(eps_pick, ls='--', lw=1.5, label=r'$\epsilon_0$')
    ax.plot([eps_pick], [G_pick], 'o', ms=7, label='Operating point')

    # Labels & cosmetics
    ax.set_xlabel(r'Detuning $\epsilon$ (eV)')
    ax.set_ylabel('Conductance G (S)')
    ax.set_title('Coulomb Peak with Operating Point')
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax




# plot coulomb peak and operating point with state separation 

def _coulomb_G(eps, g0, eps_w):
    x = 2.0 * eps / eps_w
    return g0 / np.cosh(x)**2

def plot_coulomb_peak_minimal(params_coulomb_peak,
                              delta_eps=None,        # eV (gap between charge states)
                              eps_noise=None,        # eV array: ε(t) - ε0
                              span=3.0, points=800,
                              ax=None):
    """
    Minimal, uncluttered Coulomb-peak plot:
      • Peak curve
      • Operating point
      • (optional) two state markers at ε0 ± Δε/2
      • (optional) a single thin ε-noise trajectory line

    Returns (ax, info) with info['eps0','G0','DeltaG'].
    """
    g0       = float(params_coulomb_peak['g0'])
    eps_w    = float(params_coulomb_peak['eps_width'])
    eps0_rel = float(params_coulomb_peak['eps0'])
    eps0     = eps0_rel * eps_w

    # Peak curve
    eps = np.linspace(-span * eps_w, span * eps_w, points)
    G   = _coulomb_G(eps, g0, eps_w)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))

    ax.plot(eps, G, lw=2.0, label='Coulomb peak')

    # Operating point
    G0 = _coulomb_G(eps0, g0, eps_w)
    ax.axvline(eps0, ls='--', lw=1.0, color='k', alpha=0.5)
    ax.plot([eps0], [G0], 'o', ms=6, color='k', label='Operating point')

    # Optional: two-state markers (no extra lines/annotations)
    DeltaG = None
    if delta_eps is not None and delta_eps > 0:
        eA = eps0 - 0.5 * delta_eps
        eB = eps0 + 0.5 * delta_eps
        GA = _coulomb_G(eA, g0, eps_w)
        GB = _coulomb_G(eB, g0, eps_w)
        DeltaG = abs(GB - GA)
        ax.plot([eA, eB], [GA, GB], 's', ms=5, label='States')

    # Optional: ε-noise trajectory as a single thin line
    if eps_noise is not None:
        eps_path = eps0 + np.asarray(eps_noise)
        G_path   = _coulomb_G(eps_path, g0, eps_w)
        ax.plot(eps_path, G_path, lw=1.2, alpha=0.8, label='ε-noise path')

    # Minimal cosmetics
    ax.set_xlabel(r'Detuning $\epsilon$ (eV)')
    ax.set_ylabel('Conductance G (S)')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='best', frameon=False)

    info = dict(eps0=eps0, G0=G0, DeltaG=DeltaG)
    return ax, info


def plot_coulomb_peak_inset(params_coulomb_peak,
                            ax=None,
                            delta_eps=None,     # eV (gap between charge states)
                            eps_noise=None,     # array in eV: ε(t) - ε0
                            span=2.0,           # x-range in units of eps_width
                            points=400,
                            color='0.1',        # grayscale for print-friendly insets
                            noise_color='0.3',
                            peak_lw=1.2,
                            noise_lw=0.8,
                            op_ms=3,
                            state_ms=3,
                            hide_axes=True):
    """
    Minimal Coulomb-peak insert:
      - Peak curve
      - Operating point
      - (optional) two state markers at ε0 ± Δε/2
      - (optional) ε-noise trace along the peak

    Returns the Axes.
    """
    g0    = float(params_coulomb_peak['g0'])
    eps_w = float(params_coulomb_peak['eps_width'])
    eps0  = float(params_coulomb_peak['eps0']) * eps_w

    # Peak
    eps = np.linspace(-span * eps_w, span * eps_w, points)
    G   = _coulomb_G(eps, g0, eps_w)

    if ax is None:
        fig, ax = plt.subplots(figsize=(2.6, 2.0))

    ax.plot(eps, G, lw=peak_lw, color=color)

    # Operating point
    G0 = _coulomb_G(eps0, g0, eps_w)
    ax.plot([eps0], [G0], marker='o', ms=op_ms, color=color)

    # Optional: two-state markers
    if delta_eps is not None and delta_eps > 0:
        eA = eps0 - 0.5 * delta_eps
        eB = eps0 + 0.5 * delta_eps
        GA = _coulomb_G(eA, g0, eps_w)
        GB = _coulomb_G(eB, g0, eps_w)
        ax.plot([eA, eB], [GA, GB], marker='s', ls='none', ms=state_ms, color=color, alpha=0.9)

    # Optional: ε-noise path
    if eps_noise is not None:
        epath = eps0 + np.asarray(eps_noise)
        Gpath = _coulomb_G(epath, g0, eps_w)
        ax.plot(epath, Gpath, lw=noise_lw, color=noise_color, alpha=0.75)

    # Inset-style axes
    if hide_axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    else:
        ax.tick_params(length=2, labelsize=8)
    ax.margins(x=0.03, y=0.05)
    return ax


def plot_minimal_coulomb_peak_inset(ax, params, span=2.5, points=300):
    g0 = float(params['g0'])
    eps_w = float(params['eps_width'])
    eps0_rel = float(params['eps0'])
    eps = np.linspace(-span*eps_w, span*eps_w, points)
    G = g0 / np.cosh(2.0*eps/eps_w)**2
    ax.plot(eps, G, lw=1.2, color='black')
    # mark operating point
    eps_pick = eps0_rel * eps_w
    G_pick = g0 / np.cosh(2.0*eps_pick/eps_w)**2
    ax.plot([eps_pick], [G_pick], 'o', ms=3, color='black')
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[['top','right','left','bottom']].set_linewidth(0.8)


def plot_qubit_array(dot_positions, sensor_positions, ax=None):
    """
    Minimal insert-style plot of qubit array with sensors and dots.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3), dpi=120)

    # Plot dots (qubits)
    ax.scatter(dot_positions[:, 0], dot_positions[:, 1], 
               c='red', s=90, label='Qubits')
    # Plot sensors
    ax.scatter(sensor_positions[:, 0], sensor_positions[:, 1], 
               c='blue', s=100, marker='s', label='Sensors')

    # Labels
    for i, pos in enumerate(dot_positions):
        ax.text(pos[0]+5, pos[1]+5, f'D{i}', fontsize=10, color='red')
    for i, pos in enumerate(sensor_positions):
        ax.text(pos[0]+5, pos[1]+5, f'S{i}', fontsize=10, color='blue')

    # Add a margin based on both dots and sensors so asymmetric layouts
    # (for example one sensor above and one below) are fully visible.
    all_positions = np.vstack([dot_positions, sensor_positions])
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()

    margin_x = 0.2 * (x_max - x_min) if x_max > x_min else 20
    margin_y = 0.2 * (y_max - y_min) if y_max > y_min else 20

    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)
    ax.set_aspect('equal', adjustable='box')

    # Clean look: no ticks/labels, keep frame
    ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor("black")

    ax.set_facecolor("white")

    return ax

def plot_qubit_array_insert(dot_positions, sensor_positions, ax=None):
    """
    Minimal insert-style plot of qubit array with sensors and dots.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3), dpi=120)

    # Plot dots (qubits)
    ax.scatter(dot_positions[:, 0], dot_positions[:, 1], 
               c='red', s=70, label='Qubits')
    # Plot sensors
    ax.scatter(sensor_positions[:, 0], sensor_positions[:, 1], 
               c='blue', s=80, marker='s', label='Sensors')

    # Add a margin based on both dots and sensors so inset layouts stay centered
    # even when sensors extend beyond the dot array.
    all_positions = np.vstack([dot_positions, sensor_positions])
    x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
    y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()

    margin_x = 0.2 * (x_max - x_min) if x_max > x_min else 20
    margin_y = 0.2 * (y_max - y_min) if y_max > y_min else 20

    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)
    ax.set_aspect('equal', adjustable='box')

    # Clean look: no ticks/labels, keep frame
    ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor("black")

    ax.set_facecolor("white")


def plot_qubit_layout_variants(layout_variants, C0=1.0, alpha=1.0, beta=0.1,
                               figsize=None, figure_title='Layout variants',
                               ncols=None):
    """
    Plot several dot/sensor layouts and return simple coupling summaries.

    Parameters
    ----------
    layout_variants : dict
        Mapping ``name -> {'dots': np.ndarray, 'sensors': np.ndarray}``.
    C0, alpha, beta : float
        Geometric capacitance-model parameters passed to
        ``GeometricQuantumDotSystem``.
    figsize : tuple or None
        Matplotlib figure size. If omitted, a size is chosen automatically
        from the number of layout panels.
    figure_title : str
        Figure title.
    ncols : int or None
        Number of subplot columns. If omitted, a compact grid is chosen
        automatically.

    Returns
    -------
    fig, axes, summary_rows
        ``summary_rows`` contains tuples of
        ``(name, avg_dot_dot_coupling, avg_dot_sensor_coupling, max_coupling)``.
    """
    from .quantum_dot_system import GeometricQuantumDotSystem

    n_layouts = len(layout_variants)
    if n_layouts == 0:
        raise ValueError("layout_variants must contain at least one layout.")

    if ncols is None:
        ncols = min(3, n_layouts)
    ncols = max(1, min(ncols, n_layouts))
    nrows = int(np.ceil(n_layouts / ncols))

    if figsize is None:
        figsize = (4.0 * ncols, 4.0 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor('#f7f7fb')
    axes_array = np.atleast_1d(axes).ravel()

    all_positions = [
        np.vstack([np.asarray(layout['dots']), np.asarray(layout['sensors'])])
        for layout in layout_variants.values()
    ]
    global_x_min = min(pos[:, 0].min() for pos in all_positions)
    global_x_max = max(pos[:, 0].max() for pos in all_positions)
    global_y_min = min(pos[:, 1].min() for pos in all_positions)
    global_y_max = max(pos[:, 1].max() for pos in all_positions)

    x_span = global_x_max - global_x_min
    y_span = global_y_max - global_y_min
    margin_x = 0.2 * x_span if x_span > 0 else 20
    margin_y = 0.2 * y_span if y_span > 0 else 20

    summary_rows = []

    for ax, (name, layout) in zip(axes_array, layout_variants.items()):
        variant = GeometricQuantumDotSystem(
            layout['dots'],
            layout['sensors'],
            C0=C0,
            alpha=alpha,
            beta=beta,
        )
        plot_qubit_array(layout['dots'], layout['sensors'], ax=ax)
        ax.set_xlim(global_x_min - margin_x, global_x_max + margin_x)
        ax.set_ylim(global_y_min - margin_y, global_y_max + margin_y)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(name, fontsize=11, pad=8)
        info = variant.get_coupling_info()
        summary_rows.append(
            (name, info['avg_dot_dot_coupling'], info['avg_dot_sensor_coupling'], info['max_coupling'])
        )

    for ax in axes_array[n_layouts:]:
        ax.set_visible(False)

    fig.suptitle(figure_title, fontsize=14, y=1.03)
    return fig, axes, summary_rows


def plot_geometry_coupling_summary(geo_system, dot_positions=None, sensor_positions=None,
                                   figsize=(12.8, 4.9),
                                   title='Geometry-driven quantum-dot couplings'):
    """
    Plot a geometric layout next to |Cdd| and |Cds| heatmaps.

    Parameters
    ----------
    geo_system : GeometricQuantumDotSystem
        Geometry-based system containing ``Cdd`` and ``Cds``.
    dot_positions, sensor_positions : np.ndarray or None
        Optional explicit positions. If omitted, positions are taken from
        ``geo_system``.
    figsize : tuple
        Figure size.
    title : str
        Figure title.

    Returns
    -------
    fig, (ax_layout, ax_cdd, ax_cds)
    """
    if dot_positions is None:
        dot_positions = geo_system.dot_positions
    if sensor_positions is None:
        sensor_positions = geo_system.sensor_positions

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor('#f7f7fb')
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 0.9])

    ax_layout = fig.add_subplot(gs[0, 0])
    plot_qubit_array(dot_positions, sensor_positions, ax=ax_layout)
    ax_layout.set_title('Qubit and sensor layout', fontsize=12, pad=10)

    cdd_offdiag = np.abs(geo_system.Cdd[np.triu_indices(len(dot_positions), 1)])
    cdd_scale = np.max(cdd_offdiag) if np.any(cdd_offdiag) else 1.0
    cds_scale = np.max(np.abs(geo_system.Cds)) if np.any(geo_system.Cds) else 1.0

    for i in range(len(dot_positions)):
        for j in range(i + 1, len(dot_positions)):
            strength = abs(geo_system.Cdd[i, j])
            ax_layout.plot(
                [dot_positions[i, 0], dot_positions[j, 0]],
                [dot_positions[i, 1], dot_positions[j, 1]],
                color='#DC2626',
                lw=1.0 + 2.3 * strength / cdd_scale,
                alpha=0.38,
                zorder=1,
            )

    for i in range(len(dot_positions)):
        for j in range(len(sensor_positions)):
            strength = abs(geo_system.Cds[i, j])
            ax_layout.plot(
                [dot_positions[i, 0], sensor_positions[j, 0]],
                [dot_positions[i, 1], sensor_positions[j, 1]],
                color='#2563EB',
                lw=1.0 + 2.5 * strength / cds_scale,
                alpha=0.34,
                zorder=1,
            )

    ax_cdd = fig.add_subplot(gs[0, 1])
    im1 = ax_cdd.imshow(np.abs(geo_system.Cdd), cmap='Reds')
    ax_cdd.set_title('|Cdd|', fontsize=12, pad=10)
    ax_cdd.set_xticks(range(geo_system.num_dots))
    ax_cdd.set_yticks(range(geo_system.num_dots))
    ax_cdd.set_xticklabels([f'D{i}' for i in range(geo_system.num_dots)])
    ax_cdd.set_yticklabels([f'D{i}' for i in range(geo_system.num_dots)])
    for i in range(geo_system.num_dots):
        for j in range(geo_system.num_dots):
            ax_cdd.text(j, i, f'{abs(geo_system.Cdd[i, j]):.2f}',
                        ha='center', va='center', fontsize=9, color='#111827')
    fig.colorbar(im1, ax=ax_cdd, fraction=0.046, pad=0.04)

    ax_cds = fig.add_subplot(gs[0, 2])
    im2 = ax_cds.imshow(np.abs(geo_system.Cds), cmap='Blues')
    ax_cds.set_title('|Cds|', fontsize=12, pad=10)
    ax_cds.set_xticks(range(geo_system.num_sensors))
    ax_cds.set_yticks(range(geo_system.num_dots))
    ax_cds.set_xticklabels([f'S{i}' for i in range(geo_system.num_sensors)])
    ax_cds.set_yticklabels([f'D{i}' for i in range(geo_system.num_dots)])
    for i in range(geo_system.num_dots):
        for j in range(geo_system.num_sensors):
            ax_cds.text(j, i, f'{abs(geo_system.Cds[i, j]):.2f}',
                        ha='center', va='center', fontsize=9, color='#111827')
    fig.colorbar(im2, ax=ax_cds, fraction=0.06, pad=0.04)

    fig.suptitle(title, fontsize=15, y=1.03)
    return fig, (ax_layout, ax_cdd, ax_cds)



def demo_ou_noise(T=1e-6, dt=0.5e-9, sigma=2e-2, tau=1e-7, rng=None):
    """
    Generate OU noise ε(t) with std ≈ sigma (eV) and correlation time tau (s).
    Returns t, eps_noise (so ε(t) = ε0 + eps_noise).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = int(np.ceil(T / dt))
    x = np.zeros(n, dtype=float)
    alpha = np.exp(-dt / tau)
    s = np.sqrt((1 - alpha**2)) * sigma
    for i in range(1, n):
        x[i] = alpha * x[i-1] + s * rng.normal()
    t = np.arange(n) * dt
    return t, x
