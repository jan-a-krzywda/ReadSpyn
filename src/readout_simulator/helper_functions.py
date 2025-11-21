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

    ax.axis('equal')

    # Add a margin so the border looks wider in both x and y
    x_min, x_max = dot_positions[:,0].min(), dot_positions[:,0].max()
    y_min, y_max = dot_positions[:,1].min(), dot_positions[:,1].max()

    margin_x = 0.2 * (x_max - x_min) if x_max > x_min else 20
    margin_y = 0.2 * (y_max - y_min) if y_max > y_min else 20

    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)

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

    ax.axis('equal')

    # Add a margin so the border looks wider in both x and y
    x_min, x_max = dot_positions[:,0].min(), dot_positions[:,0].max()
    y_min, y_max = dot_positions[:,1].min(), dot_positions[:,1].max()

    margin_x = 0.2 * (x_max - x_min) if x_max > x_min else 20
    margin_y = 0.2 * (y_max - y_min) if y_max > y_min else 20

    ax.set_xlim(x_min - margin_x, x_max + margin_x)
    ax.set_ylim(y_min - margin_y, y_max + margin_y)

    # Clean look: no ticks/labels, keep frame
    ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # Border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor("black")

    ax.set_facecolor("white")



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
