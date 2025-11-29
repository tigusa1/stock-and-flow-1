from os import close

import io
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st
from simulation_v3 import run_simulation
import numpy as np
import pandas as pd
import sys
from animate_accumulation import make_project_activity_animation, burn_rate, set_quarterly_ticks
import datetime
from dateutil.relativedelta import relativedelta


# ---------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------
def is_debugging():
    """Checks if the current Python process is being debugged."""
    sys_breakpoint = sys.breakpointhook.__module__ != "sys"
    sys_trace = sys.gettrace() is not None
    return sys_breakpoint or sys_trace


def thousands(x, pos):
    """Formatter: show thousands."""
    return f"{x / 1000:.0f}"


def sm(y):
    """Simple smoothing: 2-point centered rolling mean."""
    return pd.Series(y).rolling(window=2, center=True).mean().bfill().ffill()


# ---------------------------------------------------------------------
# Cloud Run–safe figure rendering
# ---------------------------------------------------------------------
def show_fig(fig):
    """
    Render a matplotlib figure via PNG bytes.
    Avoids Streamlit's in-memory media store issues on Cloud Run.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------
# Reset simulation anytime a UI widget changes
# ---------------------------------------------------------------------
def reset_simulation():
    st.session_state["run_sim"] = False


# ---------------------------------------------------------------------
# UI function: builds all Streamlit widgets and returns parameter dict
# ---------------------------------------------------------------------
def render_ui():
    """
    Build all Streamlit UI and return a dict of parameters needed
    by the simulation & plotting routine.
    """

    # === DATES ===
    start_year = 2025
    end_year = 2027
    n_simulation_months = (end_year - start_year + 1) * 12 - 1

    closing_date = datetime.datetime(2025, 11, 11)
    closing_date -= relativedelta(years=0)

    seed = 3  # currently unused

    # Flag: plot individual projects (slow)
    plot_individual_projects = st.toggle(
        "Plot individual projects (requires several minutes)", 
        on_change=reset_simulation
    )

    # === DEFAULTS ===
    n_projects_yr_0 = 3400
    cash_init_0 = 4100
    max_award_0 = 1.0
    n_months_0 = 24
    reimbursement_duration_0 = 2
    future_reimbursement_duration_0 = 2
    T1_reimbursement_duration_0 = 6.0
    p_non_reimb_0 = 0.0
    del_p_non_reimb_0 = 0.0
    del_T1_non_reimb_0 = 0.0
    p_delayed_NOA_0 = 0.0
    T1_NOA_0 = 100000
    reduction_in_burn_rate_0 = 100.0
    T1_reduction_in_burn_rate_0 = 0.0
    idc_rate_0 = 55.0
    idc_2_rate_0 = 55.0
    T1_idc_2_rate_months_0 = 0

    # === Layout ===
    st.set_page_config(
        page_title="Cash Flow Simulator",
        layout="wide",
    )

    st.title("Cash Flow Simulation Dashboard")

    col1, col2, col3 = st.columns([1, 1, 3])
    # -----------------------------------------------------------------
    # BASIC SETTINGS
    # -----------------------------------------------------------------
    with col1:
        with st.expander("Basic Settings", expanded=True):
            cash_init = st.slider(
                "Initial cash balance ($MM)", 0, 8000, cash_init_0,
                step=100, on_change=reset_simulation
            )
            cash_init *= 1000

            # Use defaults for now (sliders can be re-enabled later)
            n_projects_yr = n_projects_yr_0
            max_award = max_award_0 * 1000

            start_year = st.slider(
                "Start year", 2018, 2026, start_year, 1,
                on_change=reset_simulation
            )
            end_year = st.slider(
                "End year", 2025, 2030, end_year, 1,
                on_change=reset_simulation
            )
            n_simulation_months = (end_year - start_year + 1) * 12 - 1

    # -----------------------------------------------------------------
    # PROJECTED NEW AWARDS + REIMBURSEMENT DELAYS
    # -----------------------------------------------------------------
    with col2:
        with st.expander("Projected New Awards", expanded=True):
            new_awards_pct = (
                st.slider(
                    "New awards (% of historical rate)",
                    0.0, 150.0, 100.0, 5.0,
                    on_change=reset_simulation
                ) / 100.0
            )

        with st.expander("Reimbursement Delays", expanded=True):
            reimbursement_duration = st.slider(
                "Current delay (weeks)",
                1, 10, reimbursement_duration_0,
                on_change=reset_simulation
            )
            future_reimbursement_duration = st.slider(
                "Future delay (weeks)",
                1, 10, future_reimbursement_duration_0,
                on_change=reset_simulation
            )
            T1_reimbursement_duration = (
                st.slider(
                    "Time of future delay (months)",
                    0.0, 12.0, T1_reimbursement_duration_0, 1.0,
                    on_change=reset_simulation
                ) * 52 / 12
                + reimbursement_duration
            )

    # -----------------------------------------------------------------
    # PROJECT-LEVEL UI
    # -----------------------------------------------------------------
    with col1:
        with st.expander(
            "Pending Projects That Are Not Awarded",
            expanded=True,
        ):
            p_non_reimb = (
                st.slider(
                    "Current probability (%)",
                    0.0, 25.0, p_non_reimb_0, 1.0,
                    on_change=reset_simulation
                ) / 100.0
            )
            del_p_non_reimb = (
                st.slider(
                    "Future change (%)",
                    0.0, 75.0, del_p_non_reimb_0, 1.0,
                    on_change=reset_simulation
                ) / 100.0
            )
            del_T1_non_reimb = (
                st.slider(
                    "Time of change (months)",
                    0.0, 12.0, del_T1_non_reimb_0, 1.0,
                    on_change=reset_simulation
                ) * 52 / 12
                + reimbursement_duration
            )

        with st.expander("Reduced Expenditures (Pending Projects)", expanded=True):
            reduction_in_burn_rate = (
                st.slider(
                    "Reduced spending rate (%)",
                    0.0, 100.0, reduction_in_burn_rate_0, 1.0,
                    on_change=reset_simulation
                ) / 100.0
            )
            T1_reduction_in_burn_rate = (
                st.slider(
                    "Time of reduced spending (months)",
                    0.0, 12.0, T1_reduction_in_burn_rate_0, 1.0,
                    on_change=reset_simulation
                ) * 52 / 12
                + reimbursement_duration
            )

        p_delayed_NOA = p_delayed_NOA_0
        T1_NOA = T1_NOA_0

    # -----------------------------------------------------------------
    # IDC UI
    # -----------------------------------------------------------------
    with col2:
        with st.expander("IDC", expanded=True):
            idc_rate = (
                st.slider(
                    "Current IDC (%)",
                    0.0, 80.0, idc_rate_0, 0.5,
                    on_change=reset_simulation
                ) / 100.0
            )
            idc_2_rate = (
                st.slider(
                    "Revised IDC (%)",
                    0.0, 80.0, idc_2_rate_0, 0.5,
                    on_change=reset_simulation
                ) / 100.0
            )
            T1_idc_2_rate_months = st.slider(
                "Time of revised IDC (months)",
                0, 50, T1_idc_2_rate_months_0, 1,
                on_change=reset_simulation
            )

    # PROJECT COUNT (computed)
    P_start = 0
    P_duration = 52
    T = round(n_simulation_months * 52 / 12)
    n_projects = round(n_projects_yr / 12 * (T - P_duration))

    return dict(
        start_year=start_year,
        end_year=end_year,
        n_simulation_months=n_simulation_months,
        closing_date=closing_date,
        plot_individual_projects=plot_individual_projects,
        cash_init=cash_init,
        max_award=max_award,
        n_projects=n_projects,
        reimbursement_duration=reimbursement_duration,
        future_reimbursement_duration=future_reimbursement_duration,
        T1_reimbursement_duration=T1_reimbursement_duration,
        p_non_reimb=p_non_reimb,
        del_p_non_reimb=del_p_non_reimb,
        del_T1_non_reimb=del_T1_non_reimb,
        p_delayed_NOA=p_delayed_NOA,
        T1_NOA=T1_NOA,
        reduction_in_burn_rate=reduction_in_burn_rate,
        T1_reduction_in_burn_rate=T1_reduction_in_burn_rate,
        idc_rate=idc_rate,
        idc_2_rate=idc_2_rate,
        T1_idc_2_rate_months=T1_idc_2_rate_months,
        new_awards_pct=new_awards_pct,
        P_start=P_start,
        P_duration=P_duration,
        col3=col3,
    )
# ---------------------------------------------------------------------
# Simulation + plotting function
# ---------------------------------------------------------------------
def run_simulation_and_capture(params):

    start_year = params["start_year"]
    end_year = params["end_year"]
    n_simulation_months = params["n_simulation_months"]
    closing_date = params["closing_date"]
    plot_individual_projects = params["plot_individual_projects"]
    cash_init = params["cash_init"]
    max_award = params["max_award"]
    n_projects = params["n_projects"]
    reimbursement_duration = params["reimbursement_duration"]
    future_reimbursement_duration = params["future_reimbursement_duration"]
    T1_reimbursement_duration = params["T1_reimbursement_duration"]
    p_non_reimb = params["p_non_reimb"]
    del_p_non_reimb = params["del_p_non_reimb"]
    del_T1_non_reimb = params["del_T1_non_reimb"]
    p_delayed_NOA = params["p_delayed_NOA"]
    T1_NOA = params["T1_NOA"]
    reduction_in_burn_rate = params["reduction_in_burn_rate"]
    T1_reduction_in_burn_rate = params["T1_reduction_in_burn_rate"]
    idc_rate = params["idc_rate"]
    idc_2_rate = params["idc_2_rate"]
    T1_idc_2_rate_months = params["T1_idc_2_rate_months"]
    new_awards_pct = params["new_awards_pct"]
    P_start = params["P_start"]
    P_duration = params["P_duration"]
    col3 = params["col3"]

    # Derived timings
    T = round(n_simulation_months * 52 / 12)
    P_end = T
    d = reimbursement_duration
    T1_idc_2_rate = T1_idc_2_rate_months * 52 / 12 + reimbursement_duration

    del_T_p_non_reimb = (del_T1_non_reimb, del_p_non_reimb)
    del_T_p_idc_rate = (T1_idc_2_rate, idc_2_rate)
    p_T1_delayed_NOA = (T1_NOA, p_delayed_NOA)
    reduction_T1_burn_rate = (T1_reduction_in_burn_rate, reduction_in_burn_rate)
    future_T_reimbursement_delay = (
        T1_reimbursement_duration,
        future_reimbursement_duration,
    )

    # -----------------------------
    # RUN SIMULATION
    # -----------------------------
    (
        cash_history,
        non_reimb1,
        non_reimb2,
        idc_log,
        idc_2_log,
        inst_paid_log,
        total_avg_reimbursement,
        total_spend_reimbursable,
        total_spend_non_reimbursable,
        burns,
        burns_BL,
        projects,
        cash,
        spend_by_project,
        reimbursement_by_project,
        proj_types,
    ) = run_simulation(
        cash_init,
        p_non_reimb,
        n_projects,
        max_award,
        idc_rate,
        future_T_reimbursement_delay,
        del_T_p_non_reimb,
        del_T_p_idc_rate,
        p_T1_delayed_NOA,
        reduction_T1_burn_rate,
        T=T,
        reimbursement_duration=reimbursement_duration,
        P_start=P_start,
        P_end=P_end,
        P_duration=P_duration,
        start_year=start_year,
        closing_date=closing_date,
        new_awards_pct=new_awards_pct,
    )

    # Aggregate totals
    total_spend = np.sum(spend_by_project, axis=0)
    total_reimbursement = np.sum(reimbursement_by_project, axis=0)

    # Cash balance
    cash_balance = np.zeros(T)
    cash_balance[0] = cash_init
    for i in range(T - 1):
        if i < d:
            cash_balance[i + 1] = cash_balance[i]
        else:
            cash_balance[i + 1] = (
                cash_balance[i]
                - total_spend[i]
                + total_reimbursement[i]
            )

    # -----------------------------
    # MAKE MAIN FIGURE
    # -----------------------------
    idc_cumloss = np.cumsum(np.array(idc_log[d:]) - np.array(idc_2_log[d:]))
    fig, axs = plt.subplots(2, 2, figsize=(9, 8))
    axs = axs.flatten()
    months = np.arange(len(cash_balance[d:])) / 52 * 12

    # Cash Balance
    axs[0].plot(
        months,
        sm(cash_balance[d:] - idc_cumloss),
        label="Cash Balance",
        linewidth=2,
    )
    axs[0].set_title("Cash Balance")

    # Expenditures & Revenue
    axs[2].plot(
        months,
        sm(total_reimbursement[d:]) * 52 / 12,
        label="Revenue",
        linewidth=2,
    )
    axs[2].plot(
        months,
        sm(total_spend[d:]) * 52 / 12,
        label="Expenditures",
        linewidth=2,
    )
    axs[2].set_title("Expenditures & Revenue")

    # IDC monthly
    axs[3].plot(
        months,
        sm(idc_log[d:]) * 52 / 12,
        label="IDC (base)",
        linewidth=2,
    )
    axs[3].plot(
        months,
        sm(idc_2_log[d:]) * 52 / 12,
        label=f"IDC revised",
        linewidth=2,
    )
    axs[3].set_title("IDC (monthly)")

    # Cumulative IDC loss
    axs[1].plot(
        months,
        -idc_cumloss,
        label="Cumulative IDC loss",
        linewidth=2,
    )
    axs[1].set_title("Cumulative loss of IDC")

    # Format axes
    for ax in axs:
        set_quarterly_ticks(T, start_year, ax)
        ax.set_xlabel("Quarter")
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))
        if len(ax.get_lines()) > 1:
            ax.legend()
        ax.grid(True)

    # Display figure
    if col3 is not None:
        with col3:
            fig.tight_layout()
            show_fig(fig)
    else:
        fig.tight_layout()
        show_fig(fig)

    # -----------------------------
    # BREAKDOWN BY TYPE
    # -----------------------------
    types = list(range(3))
    burns_types = np.zeros((len(types), T))
    for typ in types:
        burns_types[typ] = burns[proj_types == typ].sum(axis=0)

    reimbursement_types = np.array(
        [reimbursement_by_project[proj_types == typ].sum(axis=0) for typ in types]
    )
    spend_types = np.array(
        [spend_by_project[proj_types == typ].sum(axis=0) for typ in types]
    )

    time_marker = future_T_reimbursement_delay[0]
    Ts_simulation = np.arange(T)

    # -----------------------------
    # STATIC "ANIMATION" PLOT (PNG)
    # -----------------------------
    with st.spinner("Creating animation... please wait ⏳"):
        if plot_individual_projects:
            png_or_buf = make_project_activity_animation(
                Ts_simulation,
                burns,
                burns_BL,
                T,
                time_marker,
                animate=False,
                reimbursement_duration=d,
                start_year=start_year,
                closing_date=closing_date,
            )
        else:
            png_or_buf = make_project_activity_animation(
                Ts_simulation,
                reimbursement_types,
                spend_types,
                T,
                time_marker,
                animate=False,
                reimbursement_duration=d,
                start_year=start_year,
            )

    # THIS LINE FIXES THE BROKEN IMAGE ISSUE
    # st.image(png_or_buf, use_container_width=True)
    # Convert the main figure to PNG bytes
    fig_buf = io.BytesIO()
    fig.savefig(fig_buf, format="png", dpi=150, bbox_inches="tight")
    fig_buf.seek(0)

    # Return byte buffers, not figure objects
    return fig_buf, png_or_buf

# ---------------------------------------------------------------------
# Streamlit script entry point
# ---------------------------------------------------------------------
# Build UI and get parameters
# Initialize session flag to control simulation execution
params = render_ui()

if "run_sim" not in st.session_state:
    st.session_state["run_sim"] = False
if "last_fig" not in st.session_state:
    st.session_state["last_fig"] = None
if "last_png" not in st.session_state:
    st.session_state["last_png"] = None

# Clicking the button triggers a new simulation
if st.button("Run simulation"):
    st.session_state["run_sim"] = True

if st.session_state["run_sim"]:
    fig_buf, png_or_buf = run_simulation_and_capture(params)
    st.session_state["last_fig"] = fig_buf
    st.session_state["last_png"] = png_or_buf
    # Once results are stored, we can turn run_sim off so sliders don't auto-run
    st.session_state["run_sim"] = False

# Always display the most recent results (if any)
if st.session_state["last_fig"] is not None:
    st.image(st.session_state["last_fig"], use_column_width=True)
    # Render the figure via BytesIO
    # buf = io.BytesIO()
    # st.session_state["last_fig"].savefig(buf, format="png", dpi=150, bbox_inches="tight")
    # buf.seek(0)
    # st.image(buf, use_container_width=True)

if st.session_state["last_png"] is not None:
    st.image(st.session_state["last_png"], use_container_width=True)

# if "run_sim" not in st.session_state:
#     st.session_state["run_sim"] = False
#
# # Run simulation only when the button is clicked
# if st.button("Run simulation"):
#     st.session_state["run_sim"] = True
#
# # After clicking the button, run the simulation once
# if st.session_state["run_sim"]:
#     run_simulation_and_capture(params)












