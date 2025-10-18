# animate_accumulation.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib as mpl


def make_project_activity_animation(t, burns, n_months, save_as="activity_animation.gif"):
    """
    Two-panel animation:
      1. Gantt chart with horizontal color-intensity variation by burn rate
      2. Animated stacked-area plot of active projects.
    """

    # ------------------------------------------------------------
    # Align inputs
    # ------------------------------------------------------------
    t = np.asarray(t)
    burns = np.atleast_2d(burns)
    if burns.shape[1] != len(t):
        n_months = burns.shape[1]
        t = np.arange(n_months)
    else:
        n_months = int(n_months)

    n_projects = burns.shape[0]
    vmax = np.max(burns)
    cmap_base = plt.get_cmap("tab10")
    colors = [cmap_base(i % 10) for i in range(n_projects)]

    # start / end indices
    active_mask = burns > 0
    start_idx = np.array([np.argmax(a) for a in active_mask])
    end_idx = np.array([len(a) - np.argmax(a[::-1]) - 1 for a in active_mask])

    # normalize burn rates for intensity mapping
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax)

    # ------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------
    fig, (ax2, ax3, ax4) = plt.subplots(
        3, 1, figsize=(9, 10),
        gridspec_kw={"height_ratios": [1, 2, 2]},
        constrained_layout=True,
    )

    # ------------------------------------------------------------
    # GANTT PANEL: horizontal intensity shading
    # ------------------------------------------------------------
    # --- Panel 1: color-coded, intensity-shaded Gantt ---
    ax2.set_xlim(0, n_months)
    ax2.set_ylim(0, n_projects)
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Project Index")
    ax2.set_title("Gantt Chart: Project Activity")

    # normalize each project's burn profile to 0–1
    norm_burns = burns / np.maximum(burns.max(axis=1, keepdims=True), 1e-12)

    # build RGB array for per-project color shading
    img = np.zeros((n_projects, n_months, 3))
    for i in range(n_projects):
        base_rgb = np.array(colors[i][:3])  # RGB from project color
        # blend brightness with burn intensity: low burn → pale, high → vivid
        # shade = 0.3 + 0.7 * norm_burns[i]  # avoid completely white
        # img[i] = base_rgb * shade[:, None]

        # make low burn ~ white, high burn ~ base color
        intensity = norm_burns[i]  # 0–1 normalized burn
        shade = 1.0 - 0.8 * intensity       # 1 → white, 0.2 → vivid
        img[i] = 1.0 - (1.0 - base_rgb) * (1.0 - shade[:, None])

    # display the combined RGB image
    im = ax2.imshow(
        img,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], 0, n_projects],
        interpolation="none"
    )

    vline2 = ax2.axvline(0, color="black", linestyle=":", lw=1.5)

    # optional colorbar showing relative intensity scale (0–1)
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(vmin=0, vmax=1))
    fig.colorbar(sm, ax=ax2, fraction=0.02, pad=0.02, label="Relative burn intensity")

    ax2.invert_yaxis()
    # ------------------------------------------------------------
    # STACKED-AREA PANEL
    # ------------------------------------------------------------
    ax3.set_xlim(0, n_months)
    ax3.set_ylim(0, vmax * n_projects * 0.6)
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Active Project Spending ($M/month)")
    vline3 = ax3.axvline(0, color="black", linestyle=":", lw=1.5)

    # ------------------------------------------------------------
    # Static bottom panel (final stacked-area snapshot)
    # ------------------------------------------------------------
    # --- Static final stacked area (bottom panel) ---
    ax4.set_xlim(0, n_months)
    ax4.set_ylim(0, np.max(burns.sum(axis=0)) * 1.2)
    ax4.set_xlabel("Month")
    ax4.set_ylabel("Final Spending ($M/month)")
    ax4.set_title("Final Stacked Area (End of Period)")

    active_idx_final = np.argsort(start_idx)
    final_colors = [colors[i] for i in active_idx_final]
    ax4.stackplot(t, *burns[active_idx_final], alpha=0.7, colors=final_colors)

    # Final total dotted curve
    total_final = burns.sum(axis=0)
    final_curve, = ax4.plot(
        t, total_final, linestyle=":", lw=3, color="black", alpha=0.9
    )

    # --- Animated stacked area (middle panel) ---
    ax3.set_xlim(0, n_months)
    # use the same y-axis limits as ax4
    ax3.set_ylim(ax4.get_ylim())
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Active Project Spending ($M/month)")
    vline3 = ax3.axvline(0, color="black", linestyle=":", lw=1.5)

    # add an empty dotted line that will always mirror the final total (same shape as ax4)
    ax3_dotted, = ax3.plot(
        t, total_final, linestyle=":", lw=3, color="black", alpha=0.5
    )

    # ------------------------------------------------------------
    # Update function
    # ------------------------------------------------------------
    def update(month_idx):
        m = int(month_idx)
        current_time = t[m]

        # --- move Gantt time marker ---
        vline2.set_xdata([current_time, current_time])

        # --- clear & redraw stacked areas ---
        # --- Middle panel (animated stacked area) ---
        for art in list(ax3.collections):
            try:
                art.remove()
            except Exception:
                pass
        for line in [l for l in ax3.lines if l not in [vline3, ax3_dotted]]:
            line.remove()

        active_mask_now = start_idx <= m
        if np.any(active_mask_now):
            active_idx = np.where(active_mask_now)[0]
            active_idx = active_idx[np.argsort(start_idx[active_idx])]
            active_burns = burns[active_idx]
            active_colors = [colors[i] for i in active_idx]

            ax3.stackplot(t, *active_burns, alpha=0.7, colors=active_colors)

            # --- solid top line for current total ---
            total_active = active_burns.sum(axis=0)
            ax3.plot(
                t, total_active,
                linestyle="-", lw=3, color="black", alpha=0.9
            )

        # keep dotted reference (ax3_dotted) as the final total
        vline3.set_xdata([current_time, current_time])
        ax3.set_xlim(0, n_months)
        ax3.set_ylim(ax4.get_ylim())

        return [vline2, vline3]

    # ------------------------------------------------------------
    # Animate & save
    # ------------------------------------------------------------
    ani = FuncAnimation(fig, update, frames=n_months, interval=200, blit=False)
    update(0)
    fig.canvas.draw()

    try:
        writer = PillowWriter(fps=5)
        ani.save(save_as, writer=writer)
        print(f"✅ Animation saved → {save_as}")
    except Exception as e:
        print(f"⚠️ Animation writer failed ({type(e).__name__}: {e}); saving static fallback.")
        fallback = save_as.replace(".gif", "_fallback.png")
        fig.savefig(fallback)
        save_as = fallback
    finally:
        plt.close(fig)

    return save_as