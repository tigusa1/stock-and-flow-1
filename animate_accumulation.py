# animate_accumulation.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib as mpl
import datetime
import calendar
import pandas as pd

import streamlit as st

def months_between(start, end):
    # whole months
    months = (end.year - start.year) * 12 + (end.month - start.month)

    # fractional part based on the starting month’s length
    days_in_start_month = calendar.monthrange(start.year, start.month)[1]
    fractional = (end.day - start.day) / days_in_start_month

    return months + fractional

def set_quarterly_ticks(T, start_year, ax):
    """
    Adds quarterly x-axis ticks and vertical grid lines for a plot where x is in months
    and t = 0 corresponds to January 1 of start_year.

    Year is shown only once per group (2024 Q1, Q2, Q3, Q4, 2025 Q1, ...).
    """

    # Total months on the x-axis
    total_months = T * 12 / 52
    ax.set_xlim(0, total_months)

    # Number of years we need to cover
    num_years = int(total_months // 12) + 2

    tick_positions = []
    tick_labels = []

    quarterly_offsets = [0, 3, 6, 9]
    quarterly_names = ["Q1", "Q2", "Q3", "Q4"]

    for year_idx in range(num_years):
        year = start_year + year_idx

        for q_i, (qname, offset) in enumerate(zip(quarterly_names, quarterly_offsets)):

            pos = year_idx * 12 + offset
            if pos > total_months:
                break

            # ---- Tick label ----
            if q_i == 0:
                label = f"{year} {qname}"
            else:
                label = qname

            tick_positions.append(pos)
            tick_labels.append(label)

            # ---- Vertical gridline ----
            ax.axvline(pos, color='lightgray', linewidth=0.8, linestyle='--')

    # Apply ticks
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    # Optional: standard grid turned on for horizontal lines
    ax.grid(axis='y', linestyle=':', color='gray', alpha=0.5)


def burn_rate(burns, burns_BL, reimbursement_duration, save_Excel = True):
    burns = np.atleast_2d(burns) # shape = number of projects, time
    burns_BL = np.atleast_2d(burns_BL)
    burns = burns[:,reimbursement_duration:]
    burns_BL = burns_BL[:,reimbursement_duration:]

    total_final = burns.sum(axis=0)
    total_BL_final = burns_BL.sum(axis=0)

    if save_Excel:
        # total_final is your numpy array
        df_out = pd.DataFrame({
            "row_number": np.arange(len(total_final)),  # 0, 1, 2, ..., 465
            "total_final": total_final
        })
        df_out.to_excel("main_streamlit_v3_temp.xlsx", index=False)

    return burns, burns_BL, total_final, total_BL_final

@st.cache_data
def make_project_activity_animation(Ts_simulation, burns, burns_BL, T, decline_month,
            save_as="activity_animation.gif", animate=True, reimbursement_duration=0,
            del_T_p_non_reimb=(0,0.), start_year=2023, closing_date=None):
    """
    Two-panel animation:
      1. Gantt chart with horizontal color-intensity variation by burn rate
      2. Animated stacked-area plot of active projects.
    """

    # ------------------------------------------------------------
    # Align inputs
    # ------------------------------------------------------------
    burns, burns_BL, total_final, total_BL_final = burn_rate(burns, burns_BL, reimbursement_duration, save_Excel = False)

    Ts_simulation = np.asarray(Ts_simulation)
    decline_month = decline_month - reimbursement_duration
    if burns.shape[1] != len(Ts_simulation):
        T = burns.shape[1]
        Ts_simulation = np.arange(T)
    else:
        T = int(T)

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
    if animate:
        fig, (ax3, ax2, ax4) = plt.subplots(
            3, 1, figsize=(9, 10),
            gridspec_kw={"height_ratios": [3, 2, 3]},
            constrained_layout=True,
        )
    else:
        fig, ax4 = plt.subplots(figsize=(9, 5),
            constrained_layout=True,
        )
        ax3 = plt.axes()
        ax2 = plt.axes()

    # ------------------------------------------------------------
    # GANTT PANEL: horizontal intensity shading
    # ------------------------------------------------------------
    # --- Panel 1: color-coded, intensity-shaded Gantt ---
    ax2.set_xlim(0, T)
    ax2.set_ylim(0, n_projects)
    ax2.set_xlabel("Week")
    ax2.set_ylabel("Project Index")
    ax2.set_title("Gantt Chart: Project Activity")

    # normalize each project's burn profile to 0–1
    norm_burns = burns / np.maximum(burns.max(axis=1, keepdims=True), 1e-12)

    # build RGB array for per-project color shading
    img = np.zeros((n_projects, T, 3))
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
        extent=[Ts_simulation[0], Ts_simulation[-1], 0, n_projects],
        interpolation="none"
    )

    vline2 = ax2.axvline(0, color="black", linestyle=":", lw=1.5)

    # optional colorbar showing relative intensity scale (0–1)
    # sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=plt.Normalize(vmin=0, vmax=1))
    # fig.colorbar(sm, ax=ax2, fraction=0.02, pad=0.02, label="Relative burn intensity")

    ax2.invert_yaxis()

    # ------------------------------------------------------------
    # STACKED-AREA PANEL
    # ------------------------------------------------------------
    # ax3.set_xlim(0, T*12/52)
    set_quarterly_ticks(T, start_year, ax3)

    ax3.set_ylim(0, vmax * n_projects * 0.6)
    ax3.set_xlabel("Quarter")
    ax3.set_ylabel("Spend rate ($MM/month)")
    vline3 = ax3.axvline(0, color="black", linestyle=":", lw=1.5)
    if decline_month > 0:
        # REMOVE delayed NOA
        vlineD = None
        # vlineD = ax3.axvline(decline_month*12/52, color="red", ls=":", label="Time of increased reimbursement delay")
        # ax4.axvline(decline_month*12/52, color="red", ls=":", label="Time of increased reimbursement delay")
    else:
        vlineD = None
    if del_T_p_non_reimb[1] > 0:
        ax4.axvline((del_T_p_non_reimb[0] - reimbursement_duration)*12/52, color="yellow", ls=":",
                    label="Month of increased probability \nof unawarded pending projects")
    if closing_date:
        ax4.axvline(months_between(datetime.datetime(start_year,1,1),closing_date), color="red", ls=":",
                    label="database closing date")
    # ax2.axvline(decline_month, color="red", ls=":", label="Decline starts")

    # ------------------------------------------------------------
    # Static bottom panel (final stacked-area snapshot)
    # ------------------------------------------------------------
    # --- Static final stacked area (bottom panel) ---
    # ax4.set_xlim(0, T*12/52)
    set_quarterly_ticks(T, start_year, ax4)

    # ax4.set_ylim(0, np.max(burns_BL.sum(axis=0)) * 1.05*52/12/1000)
    ax4.set_ylim(0, np.max(burns.sum(axis=0)) * 1.05*52/12/1000)
    ax4.set_xlabel("Calendar Year Quarter")
    ax4.set_ylabel("$MM / month")
    ax4.set_title("Total Project Expenditure Run Rate")
    ax3.set_title("Spending of active and past awards")

    # active_idx_final = np.argsort(start_idx)
    # final_colors = [colors[i] for i in active_idx_final]

    # Fixed order of projects (earliest start first)
    # labels = ['normal projects', 'no NOA', 'delayed NOA', 'est. new awards'] # REMOVE delayed NOA
    labels = ['normal projects', 'unawarded pending projects', 'estimated new awards']
    final_order = np.argsort(start_idx)
    if len(final_order) > len(labels):
        final_labels = []
    else:
        final_order = np.array(range(len(labels)))
        final_labels = labels
        # final_labels = [labels[i] for i in final_order]
        print(f"final_order: {final_order}")

    final_colors = [colors[i] for i in final_order]
    ax4.stackplot(Ts_simulation * 12 / 52, *burns[final_order] * 52 / 12 / 1000, alpha=0.7 + 0.3, colors=final_colors,
                  labels = final_labels)

    # Final total dotted curve
    # final_curve, = ax4.plot(
    #     t, total_final, linestyle=":", lw=3, color="red", alpha=0.9+0.1
    # )
    # final_BL_curve, = ax4.plot(
    #     t, total_BL_final, linestyle=":", lw=3, color="black", alpha=0.9+0.1
    # )

    # --- Animated stacked area (middle panel) ---
    ax3.set_xlim(0, T)
    # use the same y-axis limits as ax4
    ax3.set_ylim(ax4.get_ylim())
    ax3.set_xlabel("Week")
    ax3.set_ylabel("Spending per week ($M)")
    vline3 = ax3.axvline(0, color="black", linestyle=":", lw=1.5)

    # add an empty dotted line that will always mirror the final total (same shape as ax4)
    total_style    = dict(linestyle=":", lw=3, color="red",   alpha=0.5+0.5, label="total award")
    total_BL_style = dict(linestyle=":", lw=3, color="black", alpha=0.5+0.5,
                          label="total award without declines")
    ax3_dotted, = ax3.plot(Ts_simulation, total_final, **total_style)
    # ax4.plot(              t, total_final, **total_style)
    ax3_BL_dotted, = ax3.plot(Ts_simulation, total_BL_final, **total_BL_style)
    # ax4.plot(                 t, total_BL_final, **total_BL_style)
    # ax3.legend(fontsize=8, loc="upper right")
    handles, labels = ax4.get_legend_handles_labels()
    if labels:  # or if handles
        ax4.legend(fontsize=12, loc="lower center")

    # ------------------------------------------------------------
    # Update function
    # ------------------------------------------------------------
    def update(month_idx):
        m = int(month_idx)
        current_time = Ts_simulation[m]

        # --- move Gantt time marker ---
        vline2.set_xdata([current_time, current_time])

        # --- clear & redraw stacked areas ---
        # --- Middle panel (animated stacked area) ---
        for art in list(ax3.collections):
            try:
                art.remove()
            except Exception:
                pass
        for line in [l for l in ax3.lines if l not in [vline3, ax3_dotted,
                ax3_BL_dotted, vlineD]]:
            line.remove()

        active_mask_now = start_idx <= m
        if np.any(active_mask_now):
            # Indices of active projects in the fixed final order
            active_idx = [i for i in final_order if active_mask_now[i]]
            active_burns = burns[active_idx]
            active_colors = [final_colors[final_order.tolist().index(i)] for i in active_idx]

            # active_idx = np.where(active_mask_now)[0]
            # active_idx = active_idx[np.argsort(start_idx[active_idx])]
            # active_burns = burns[active_idx]
            # # active_colors = [colors[i] for i in active_idx] # flickering
            # # active_colors = [final_colors[active_idx_final.tolist().index(i)] for i in active_idx]
            # active_colors = final_colors[:len(active_idx)]

            ax3.stackplot(Ts_simulation, *active_burns, alpha=0.7 + 0.3, colors=active_colors)

            # # ✅ Rasterize all filled polygons for this frame SLOW IN CLOUD!
            # for art in ax3.collections:
            #     art.set_rasterized(True)

            # --- solid top line for current total ---
            total_active = active_burns.sum(axis=0)
            ax3.plot(
                Ts_simulation, total_active,
                linestyle="-", lw=3, color="black", alpha=0.9+0.1
            )

        # keep dotted reference (ax3_dotted) as the final total
        vline3.set_xdata([current_time, current_time])
        ax3.set_xlim(0, T)
        ax3.set_ylim(ax4.get_ylim())

        return [vline2, vline3]

    # ------------------------------------------------------------
    # Animate & save
    # ------------------------------------------------------------
    ani = FuncAnimation(fig, update, frames=T, interval=200, blit=False)
    # ani = FuncAnimation(fig, update, frames=t[::2], interval=200, blit=False) # NOT MUCH DIFFERENCE
    update(0)
    fig.canvas.draw()

    if animate:
        fig.delaxes(ax4)
        # writer = PillowWriter(fps=5)
        # ani.save(save_as, writer=writer)
        ani.save(save_as, writer='ffmpeg')
        print(f"✅ Animation saved → {save_as}")
        plt.close(fig)
    else:
        fig.delaxes(ax2) # Gantt
        fig.delaxes(ax3) # animation (beginning at time t = 0)
        # fig.subplots_adjust(hspace=0.3)  # tighten spacing after removing

        flag_buffer = True
        # USE BUFFER
        if flag_buffer:
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
        else:
            # USE PNG
            static_path = save_as.replace(".gif", "_static.png")
            fig.savefig(static_path, dpi=600)
            # fig.savefig(static_path, bbox_inches="tight")

        plt.close(fig)

        if flag_buffer:
            save_as = buf
            # AFTER RETURN
            # st.image(buf)
        else:
            save_as = static_path # stacked plot at end time, saved in activity_animation_static.png

    return save_as