from os import close

import io
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st
from simulation_v3 import run_simulation
from project_generator import burn_curve, generate_projects
import numpy as np
import pandas as pd
import sys
from PIL import Image
import subprocess
from animate_accumulation import make_project_activity_animation, burn_rate, set_quarterly_ticks
import datetime
from dateutil.relativedelta import relativedelta


# ---------------------------------------------------------------------
# Utility / helper functions
# ---------------------------------------------------------------------
def is_debugging():
    """Checks if the current Python process is being debugged."""
    sys_breakpoint = sys.breakpointhook.__module__ != "sys"
    sys_trace = sys.gettrace() is not None
    return sys_breakpoint or sys_trace


def is_debugging_NG():
    """Checks if the current Python process is being debugged (old version)."""
    return hasattr(sys, "gettrace") and sys.gettrace() is not None


def thousands(x, pos):
    """Formatter: show thousands instead of full value."""
    return f"{x / 1000:.0f}"


def sm(y):
    """Simple smoothing: 2-point centered rolling mean."""
    return pd.Series(y).rolling(window=2, center=True).mean().bfill().ffill()


def show_fig(fig):
    """
    Render a matplotlib figure via PNG bytes.
    This avoids Streamlit's in-memory media store issues on Cloud Run.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, use_column_width=True)
    plt.close(fig)


# ---------------------------------------------------------------------
# UI function: builds all Streamlit widgets and returns parameters
# ---------------------------------------------------------------------
def render_ui():
    """
    Build all Streamlit UI and return a dict of all parameters
    needed by the simulation and plotting routine.
    """

    # === DATES OF ANALYSIS, CLOSING DATE ===
    start_year = 2025
    end_year = 2027
    n_simulation_months = (end_year - start_year + 1) * 12 - 1

    # closing_date corresponds to the Excel file date
    closing_date = datetime.datetime(2025, 11, 11)
    closing_date -= relativedelta(years=0)
    seed = 3  # currently unused, but kept for consistency

    # === USE SAVED PARAMETERS OR DEFAULT VALUES ===
    if is_debugging():
        flag_use_saved = False
        plot_individual_projects = False
    else:
        flag_use_saved = False
        # flag_use_saved = st.toggle("Use saved values")
        plot_individual_projects = st.toggle("Plot individual projects (requires several minutes)")

    if flag_use_saved:
        n_projects_yr_0 = 3400
        cash_init_0 = 4100
        max_award_0 = 1.0
        n_months_0 = 24 + 24

        reimbursement_duration_0 = 2
        future_reimbursement_duration_0 = 2
        T1_reimbursement_duration_0 = 6.0

        p_non_reimb_0 = 10.0
        del_p_non_reimb_0 = 0.0
        del_T1_non_reimb_0 = 12.0

        p_delayed_NOA_0 = 0.0
        T1_NOA_0 = 100000
        reduction_in_burn_rate_0 = 50.0
        T1_reduction_in_burn_rate_0 = 4.0
        idc_rate_0 = 55.0
        idc_2_rate_0 = 55.0
        T1_idc_2_rate_months_0 = 12
    else:
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
        T1_idc_2_rate_months_0 = 0  # all time in weeks unless the variable name includes months

    # === DEBUGGING: SMALL NUMBER OF PROJECTS, STREAMLIT: USE SLIDER VALUES ===
    if is_debugging():
        reimbursement_duration = 2
        cash_init = 4100
        n_projects = 20
        max_award = 1.0  # $MM
        max_award *= 1000  # convert to $K
        new_awards_pct = 1.0

        p_non_reimb = 0.0
        p_delayed_NOA = 0.0
        del_T1_non_reimb = 0
        del_p_non_reimb = 0.0
        idc_rate = 0.55
        T1_idc_2_rate_months = 0
        idc_2_rate = idc_rate - 0.00

        T1_NOA = 0.0
        T1_reduction_in_burn_rate = 0.0
        reduction_in_burn_rate = 0.0
        T1_reimbursement_duration = 0.0
        future_reimbursement_duration = 2

        # P_start, P_end, P_duration disregarded if Excel file data is read, P_start is set to 0
        P_start = 0  # in weeks
        P_end = 52 * 2
        P_duration = 5  # not used if projects are read

        col3 = None  # not used in debugging
    else:
        # --- Page config & global style ---
        st.set_page_config(
            page_title="Cash Flow Simulator",
            layout="wide",
        )
        st.markdown(
            """
        <style>
        details > summary {
          font-size: 2.5em !important;
          font-weight: 600 !important;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns([1, 1, 3])

        # --- BASIC SETTINGS ---
        with col1:
            with st.expander("Basic Settings", expanded=True):
                cash_init = st.slider("Initial cash balance ($MM)", 0, 8000, cash_init_0, step=100)
                cash_init *= 1000
                # For now, use default N and award size; sliders can be reintroduced later:
                n_projects_yr = n_projects_yr_0
                max_award = max_award_0
                max_award *= 1000  # convert to $K
                start_year = st.slider("Start year", 2018, 2026, start_year, 1)
                end_year = st.slider("End year", 2025, 2030, end_year, 1)
                n_simulation_months = (end_year - start_year + 1) * 12 - 1

        # --- NEW AWARDS & REIMBURSEMENT DELAYS ---
        with col2:
            with st.expander("Projected New Awards", expanded=True):
                new_awards_pct = (
                    st.slider(
                        "New awards (percent of historical award rate)",
                        0.0,
                        150.0,
                        100.0,
                        5.0,
                    )
                    / 100.0
                )

            with st.expander("Reimbursement Delays", expanded=True):
                reimbursement_duration = st.slider(
                    "Current delay (weeks)", 1, 10, reimbursement_duration_0
                )
                future_reimbursement_duration = st.slider(
                    "Future delay (weeks)", 1, 10, future_reimbursement_duration_0
                )
                T1_reimbursement_duration = (
                    st.slider(
                        "Time of future delay (months)",
                        0.0,
                        12.0,
                        T1_reimbursement_duration_0,
                        1.0,
                    )
                    * 52
                    / 12
                    + reimbursement_duration
                )

        # --- PROJECTS ---
        with col1:
            with st.expander(
                "Pending Projects That Are Not Awarded (use start year = 2022 to see long-term effects)",
                expanded=True,
            ):
                p_non_reimb = (
                    st.slider(
                        "Current probability (%)", 0.0, 25.0, p_non_reimb_0, 1.0
                    )
                    / 100.0
                )
                del_p_non_reimb = (
                    st.slider(
                        "Future change in probability (%)",
                        0.0,
                        75.0,
                        del_p_non_reimb_0,
                        1.0,
                    )
                    / 100.0
                )
                del_T1_non_reimb = (
                    st.slider(
                        "Time of probability change (months)",
                        0.0,
                        12.0,
                        del_T1_non_reimb_0,
                        1.0,
                    )
                    * 52
                    / 12
                    + reimbursement_duration
                )

            with st.expander("Reduced Expenditures of Pending Projects", expanded=True):
                reduction_in_burn_rate = (
                    st.slider(
                        "Reduced spending rate (< 100%)",
                        0.0,
                        100.0,
                        reduction_in_burn_rate_0,
                        1.0,
                    )
                    / 100.0
                )
                T1_reduction_in_burn_rate = (
                    st.slider(
                        "Time of reduced spending (months)",
                        0.0,
                        12.0,
                        T1_reduction_in_burn_rate_0,
                        1.0,
                    )
                    * 52
                    / 12
                    + reimbursement_duration
                )

            # DON'T USE SLIDERS HERE, JUST USE DEFAULT VALUE
            p_delayed_NOA = p_delayed_NOA_0
            T1_NOA = T1_NOA_0

        # --- IDC ---
        with col2:
            with st.expander("IDC", expanded=True):
                idc_rate = (
                    st.slider(
                        "Current IDC (%)",
                        0.0,
                        80.0,
                        idc_rate_0,
                        0.5,
                    )
                    / 100.0
                )
                idc_2_rate = (
                    st.slider(
                        "Revised IDC (%)",
                        0.0,
                        80.0,
                        idc_2_rate_0,
                        0.5,
                    )
                    / 100.0
                )
                T1_idc_2_rate_months = st.slider(
                    "Time of revised IDC (months)",
                    0,
                    50,
                    T1_idc_2_rate_months_0,
                    1,
                )

        # --- Derived project count ---
        P_start = -52 + 52 * 0  # use 0 for debugging and testing
        P_duration = 52
        n_projects = round(
            n_projects_yr
            / 12
            * (n_simulation_months - P_start * 12 / 52 - P_duration * 12 / 52)
        )
        print(f"n_projects: {n_projects}")

    # Pack everything we need into params dict
    return dict(
        start_year=start_year,
        end_year=end_year,
        n_simulation_months=n_simulation_months,
        closing_date=closing_date,
        seed=seed,
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
        col3=None if is_debugging() else col3,
    )


# ---------------------------------------------------------------------
# Simulation + plotting function
# ---------------------------------------------------------------------
def run_simulation_and_plot(params):
    """
    Heavy simulation + plotting.
    Called only when the user clicks the button
