from os import close

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st
from simulation_v3 import run_simulation
from project_generator import burn_curve, generate_projects
import numpy as np
import pandas as pd
import sys
import streamlit as st
from PIL import Image
import subprocess
from animate_accumulation import make_project_activity_animation, burn_rate, set_quarterly_ticks
import datetime
from dateutil.relativedelta import relativedelta

# READ REAL PROJECTS

def is_debugging():
    """Checks if the current Python process is being debugged."""
    sys_breakpoint = sys.breakpointhook.__module__ != "sys"
    sys_trace = sys.gettrace() != None
    return sys_breakpoint or sys_trace

def is_debugging_NG():
    """Checks if the current Python process is being debugged."""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

def thousands(x, pos):
    return f"{x / 1000:.0f}"  # change labels instead of dividing by 1000

def sm(y):
    return pd.Series(y).rolling(window=2, center=True).mean().bfill().ffill()

# TYPES OF PROJECTS
#   1 normal
#   2 no NOA
#   3 delayed NOA (fixed delay)
# REDUCED BURN RATE
#   A reduction in burn rate, time of reduction (maximum weight time for NOA)
# NUMBER OF APPROVED PROJECTS
#   B reduction in approved projects, time of reduction
# REIMBURSEMENT
#   C increased delay, time of increased delay
# IDC
#   D change in rate, time of change

# --- STREAMLIT ENTRY POINT ---
def run_streamlit():
    # === DATES OF ANALYSIS, CLOSING DATE ===
    # print("gettrace():", sys.gettrace())
    start_year = 2025
    end_year = 2027
    n_simulation_months = (end_year - start_year + 1) * 12 - 1
    # closing_date corresponds to the Excel file date
    # relativedelta(years) is used to examine previous closing dates (retaining only new awards up to adjusted closing date
    closing_date = datetime.datetime(2025,11,11)
    closing_date -= relativedelta(years=0)
    # closing_date -= relativedelta(years=1)
    seed = 3

    if st.button("Run simulation"):
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
            T1_idc_2_rate_months_0 = 0 # all time in weeks unless the variable name includes months

        # === DEBUGGING: SMALL NUMBER OF PROJECTS, STREAMLIT: USE SLIDER VALUES ===
        if is_debugging():
            reimbursement_duration = 2
            cash_init = 4100
            n_projects = 20
            max_award = 1.0 # $MM
            max_award *= 1000  # convert to $K
            new_awards_pct = 1.0

            # decline_month = 10
            # decline_factor = 50/100
            # n_simulation_months = 36 + 12
            # n_simulation_months = (2026 - start_year + 1) * 12 # until the end of 2026
            # n_simulation_months = (2030 - start_year + 1) * 12 # until the end of 2030
            p_non_reimb = 0.0
            p_delayed_NOA = 0.0
            del_T1_non_reimb = 0
            del_p_non_reimb = 0.0
            idc_rate = 0.55
            T1_idc_2_rate_months = 0
            idc_2_rate = idc_rate - 0.00
            # === NEW SLIDERS ===
            T1_NOA = 0.0
            T1_reduction_in_burn_rate = 0.0
            reduction_in_burn_rate = 0.0
            T1_reimbursement_duration = 0.0
            future_reimbursement_duration = 2

            # === NEW PARAMETERS ===
            # P_start, P_end, P_duration disregarded if Excel file data is read, P_start is set to 0
            P_start = 0 # in weeks
            P_end = 52 * 2
            P_duration = 5 # not used if projects are read
        else:
            st.set_page_config(
                page_title="Cash Flow Simulator",
                layout="wide"
            )
            # GLOBAL STYLE OVERRIDES
            st.markdown("""
            <style>
            details > summary {
              font-size: 2.5em !important;
              font-weight: 600 !important;
            }
            </style>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 1, 3])  # adjust width ratio if needed
            with col1:
                with st.expander("Basic Settings", expanded=True):
                    cash_init  = st.slider("Initial cash balance ($MM)",    0, 8000, cash_init_0, step=100)
                    cash_init *= 1000
                    # n_projects = st.slider("Number of Projects",    2, 80, 2, step=5) # v2
                    # max_award  = st.slider("Max Award Amount (yearly, $100MM)",      0.1,  2.0,  1.0, step=0.1)
                    # max_award *= 1000*100/100 # convert to $K
                    # n_projects = st.slider("Number of Projects",    5, 80, 45, step=5)
                    # max_award *= 1000*10 # convert to $K
                    # n_simulation_months = st.slider("Simulation duration (months)", 2, 4, 2, help=".")
                    # DISABLE SLIDERS
                    n_projects_yr = n_projects_yr_0
                    max_award = max_award_0
                    # n_projects_yr = st.slider("Number of projects per year (1000s)", 1000, 5000, n_projects_yr_0, step=100)
                    # max_award  = st.slider("Average award ($MM/year)",      0.1,  2.0,  max_award_0, step=0.1)
                    max_award *= 1000 # convert to $K
                    start_year = st.slider("Start year", 2018, 2026, start_year, 1)
                    end_year = st.slider("End year", 2025, 2030, end_year, 1)
                    # n_simulation_months = st.slider("Simulation duration (months)", 24, 84, n_months_0, 3,
                    #                                 help=".")
                    n_simulation_months = (end_year - start_year + 1) * 12 - 1

            with col2:
                with st.expander("Projected New Awards", expanded=True):
                    new_awards_pct = st.slider("New awards (percent of historical award rate)", 0., 150., 100., 5.) / 100.

                with st.expander("Reimbursement Delays", expanded=True):
                    reimbursement_duration = st.slider("Current delay (weeks)", 1, 10, reimbursement_duration_0)
                    future_reimbursement_duration = st.slider("Future delay (weeks)", 1, 10, future_reimbursement_duration_0)
                    T1_reimbursement_duration = st.slider("Time of future delay (months)",
                                                 0.0, 12.0, T1_reimbursement_duration_0, 1.0) * 52 / 12 + \
                                                 reimbursement_duration

            # PROJECTS
            with col1:
                # st.markdown("#### Award information")  # renders like st.subheader

                # with st.expander("ℹ️ Summary of simulation procedure"):
                #     st.markdown("""
                #     A series of projects are simulated as follows:
                #     - The **total number** of projects is set by the *1st slider*.
                #     - **Durations** are uniformly distributed from 12 to 36 months.
                #     - The average **burn rate** is $100,000 per month.
                #     - **Burn rate shapes** are set by the *dropdown menu* below.
                #     - The **simulation display** begins at time 0 (months), occurring after some of the projects
                #       have started. The simulation **end time** is set by the *4th slider* below.
                #     - Project **start times** are uniformly distributed starting 36 months prior to
                #       time 0 until the simulation end time.
                #
                #     A decrease in award amounts is simulated as follows:
                #     - The *2nd slider* below sets the **month at which all awards with subsequent start times** have
                #       reduced award amounts.
                #     - The **reduced award amount**, specified as a percent of the original award amount,
                #       is specified by the *3rd slider* below.
                #     """)
                #
                # with st.expander("ℹ️ Future possible developments in the simulator"):
                #     st.markdown("""
                #     This basic simulator can be further developed in several ways:
                #     - Insert **actual awards**.  Tens of thousands can be added.  The charts can be modified
                #       to display **categories of awards** (e.g., grants, contracts, subawards) instead of
                #       individual awards.
                #     - Display the **projected trends** described in VP Heller's email.
                #     - Use **probability of award termination** instead of declining award amounts.
                #     - Convert awards to **equivalent flows**.
                #     - Include **endogenous feedback**, such as the effects of reduced spending on
                #       staff/faculty (through RIF) on future awards.
                #     """)
                #
                with st.expander("Pending Projects That Are Not Awarded (use start year = 2022 to see long-term effects)", expanded=True):
                    p_non_reimb = st.slider("Current probability (%)", 0.0, 25., p_non_reimb_0, 1.0) / 100
                    del_p_non_reimb = st.slider("Future change in probability (%)",
                                                0.0, 75.0, del_p_non_reimb_0, 1.0) / 100
                    del_T1_non_reimb = st.slider("Time of probability change (months)",
                                                 0.0, 12.0, del_T1_non_reimb_0, 1.0) * 52 / 12 + \
                        reimbursement_duration

                with st.expander("Reduced Expenditures of Pending Projects", expanded=True):
                    reduction_in_burn_rate = st.slider("Reduced spending rate (< 100%)",
                                                       0.0, 100.0, reduction_in_burn_rate_0, 1.0) / 100
                    T1_reduction_in_burn_rate = st.slider("Time of reduced spending (months)",
                                                          0.0, 12.0, T1_reduction_in_burn_rate_0, 1.0) * 52 / 12 + \
                        reimbursement_duration

                # DON'T USE SLIDERS HERE, JUST USE DEFAULT VALUE
                p_delayed_NOA = p_delayed_NOA_0
                T1_NOA = T1_NOA_0

                # with st.expander("Delays in NOA", expanded=True):
                #     p_delayed_NOA = st.slider("Probability of delayed NOA", 0.0, 100.0, p_delayed_NOA_0, 1.0) / 100
                #     T1_NOA = st.slider("Delay in NOA (months)", 0.0, 12.0, T1_NOA_0, 0.5) * 52 / 12
                #     reduction_in_burn_rate = st.slider("Reduced spending rate (< 100%) due to delayed NOA (%)",
                #                                        0.0, 100.0, reduction_in_burn_rate_0, 1.0) / 100
                #     T1_reduction_in_burn_rate = st.slider("Time of reduced spending (months)",
                #                                           0.0, 12.0, T1_reduction_in_burn_rate_0, 0.5) * 52 / 12 + \
                #                                 reimbursement_duration

            with col2:
                with st.expander("IDC", expanded=True):
                    idc_rate = st.slider("Current IDC (%)", 0.0, 80.0, idc_rate_0, 0.5) / 100
                    idc_2_rate = st.slider("Revised IDC (%)", 0.0, 80.0, idc_2_rate_0, 0.5) / 100
                    T1_idc_2_rate_months = st.slider("Time of revised IDC (months)", 0, 50, T1_idc_2_rate_months_0, 1)

            P_start = -52 + 52*0 # use 0 for debugging and testing
            P_duration = 52
            n_projects = round(n_projects_yr / 12 * (n_simulation_months - P_start * 12 / 52 - P_duration * 12 / 52))
            print(f"n_projects: {n_projects}")

        T1_idc_2_rate = T1_idc_2_rate_months * 52 / 12 + reimbursement_duration
        del_T_p_non_reimb = (del_T1_non_reimb, del_p_non_reimb)
        del_T_p_idc_rate = (T1_idc_2_rate, idc_2_rate)
        p_T1_delayed_NOA = (T1_NOA, p_delayed_NOA)
        reduction_T1_burn_rate = (T1_reduction_in_burn_rate, reduction_in_burn_rate)
        future_T_reimbursement_delay = (T1_reimbursement_duration, future_reimbursement_duration)

        T = round(n_simulation_months * 52 / 12)
        P_end = T
        print(f"P_start: {P_start}, P_end: {P_end}")

        # === RUN SIMULATION OF PROJECTS ===
        (cash_history, non_reimb1, non_reimb2, idc_log, idc_2_log, inst_paid_log, total_avg_reimbursement,
         total_spend_reimbursable, total_spend_non_reimbursable,
         burns, burns_BL, projects, cash, spend_by_project, reimbursement_by_project,
         proj_types) = run_simulation(  # v2 return np grants arrays
            cash_init, p_non_reimb, n_projects, max_award, idc_rate,
            future_T_reimbursement_delay,
            del_T_p_non_reimb, del_T_p_idc_rate, p_T1_delayed_NOA, reduction_T1_burn_rate,
            T=T,  # v2
            reimbursement_duration=reimbursement_duration,
            P_start=P_start, P_end=P_end, P_duration=P_duration, start_year=start_year, closing_date=closing_date,
            new_awards_pct=new_awards_pct,
        )
        total_spend = np.sum(spend_by_project, axis=0)
        total_reimbursement = np.sum(reimbursement_by_project, axis=0)
        cash_balance = np.zeros(T)
        cash_balance[0] = cash_init

        d = reimbursement_duration # used for convenience
        for Ts_simulation in range(T - 1):
            if Ts_simulation < d:
                # reimbursements are for expenditures at time reimbursement_duration prior
                cash_balance[Ts_simulation + 1] = cash_balance[Ts_simulation]
            else:
                cash_balance[Ts_simulation + 1] = cash_balance[Ts_simulation] - total_spend[Ts_simulation] + total_reimbursement[Ts_simulation]
            # print(f"{Ts_simulation:.0f}: {cash_balance[Ts_simulation]:.0f}, {total_spend[Ts_simulation]:.0f}, {total_reimbursement[Ts_simulation]:.0f}")

        # === PLOTS ===
        flag_original_plot = False
        if flag_original_plot:
            fig, ax = plt.subplots(1, 1, figsize=(9, 8))
            axs = [ax,]
            axs[0].plot(cash_history, label="Cash Balance", linewidth=2)
            axs[0].plot(non_reimb1, label="Receivable (Reimbursable Projects)", linestyle="--")
            axs[0].plot(non_reimb2, label="Receivable (Non-reimbursable Projects)", linestyle=":")
        else:
            idc_cumloss = np.cumsum(np.array(idc_log[d:]) - np.array(idc_2_log[d:]))
            fig, axs = plt.subplots(2, 2, figsize=(9, 8))
            axs = axs.flatten()
            months = np.arange(len(cash_balance[d:]))/52*12
            axs[0].plot(months, sm(cash_balance[d:] - idc_cumloss), label="Cash Balance", linewidth=2)
            axs[0].set_title("Cash Balance")
            axs[2].plot(months, sm(total_reimbursement[d:])*52/12, label="Revenue", linewidth=2)
            axs[2].plot(months, sm(total_spend[d:])*52/12, label="Expenditures", linewidth=2)
            axs[2].set_title("Expenditures & Revenue")
            axs[3].plot(months, sm(idc_log[d:])*52/12, label="IDC (55%)", linewidth=2)
            axs[3].plot(months, sm(idc_2_log[d:])*52/12, label=f"IDC ({idc_2_rate*100:.0f}% at {T1_idc_2_rate_months:.0f} months)", linewidth=2)
            axs[3].set_title("IDC (monthly)")
            axs[1].plot(months, -idc_cumloss, label="Cumulative loss of IDC",
                        linewidth=2)
            axs[1].set_title("Cumulative loss of IDC")
            axs[0].set_ylabel("Amount ($MM)")
            axs[1].set_ylabel("Amount ($MM)")
            axs[2].set_ylabel("Spend rate ($MM/month)")
            axs[3].set_ylabel("Spend rate ($MM/month)")

        for ax in axs:
            set_quarterly_ticks(T, start_year, ax)
            ax.set_xlabel("Quarter")
            # ax.set_xlabel("Time (months)")
            # Format Y-axis ticks with commas
            # ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax.yaxis.set_major_formatter(FuncFormatter(thousands))
            if len(ax.get_lines()) > 1:
                ax.legend()
            ax.grid(True)

        if not is_debugging():
            with col3:
                # st.subheader("Cash and Receivable Balances Over Time")
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
        else:
            plt.tight_layout()
            plt.show()

        # === MODIFICATION: DON'T REGENERATE burns, burns_BL
        # Generate data
        Ts_simulation = np.arange(T)
        # burns,burns_BL = generate_projects(Ts_simulation, n_grants, decline_month, decline_factor, T,
        #                                    seed=seed, shape="constant")

        # === SAVE BURN RATE IN EXCEL ===
        burn_rate(burns, burns_BL, reimbursement_duration)

        print(f"burns {np.shape(burns)}")
        print(f"reimbursement_by_project {np.shape(reimbursement_by_project)}")
        print(f"spend_by_project {np.shape(spend_by_project)}")
        print(f"proj_types {np.shape(proj_types)}")

        # REPLACE burns WITH reimbursement_types, spend_types
        # REMOVE delayed NOA
        types = list(range(3)) # ***** REPLACE WITH ACTUAL NUMBER OF TYPES *****
        # types = np.unique(proj_types)
        burns_types = np.zeros((len(types),T))
        for typ in types:
            burns_types[typ] = burns[proj_types == typ].sum(axis=0)
            # burns_types = np.array([burns[proj_types == typ].sum(axis=0) for typ in types])

        reimbursement_types = np.array([reimbursement_by_project[proj_types == typ].sum(axis=0) for typ in types])
        spend_types = np.array([spend_by_project[proj_types == typ].sum(axis=0) for typ in types])

        time_marker = future_T_reimbursement_delay[0]

        if plot_individual_projects:
            png_path = make_project_activity_animation(Ts_simulation, burns, burns_BL, T, time_marker, animate=False,
                                                       reimbursement_duration=d, start_year=start_year,
                                                       closing_date=closing_date)  # DEBUG
            # png_path = make_project_activity_animation(Ts_simulation, reimbursement_by_project, burns, T, time_marker,
            #                                            animate=False, reimbursement_duration=d)  # DEBUG
        else:
            png_path = make_project_activity_animation(Ts_simulation, reimbursement_types, spend_types, T, time_marker,
                                                       animate=False, reimbursement_duration=d,
                                                       start_year=start_year)  # DEBUG

        if is_debugging():
            print("DEBUG MODE")
            import matplotlib.image as mpimg
            fig, ax = plt.subplots(figsize=(9, 5)) # make figsize same as in make_project_activity_animation()
            img = mpimg.imread(png_path)
            ax.imshow(img)
            ax.axis('off')
            plt.show()
            # gif_path = make_project_activity_animation(Ts_simulation, burns, burns_BL, T, decline_month) # DEBUG
        else:
            with col3:
                # Expenditures compared with spending
                # st.image(png_path, width="stretch")
                # ANIMATION
                with st.spinner("Creating animation... please wait ⏳"):
                    if plot_individual_projects:
                        # NOT BURN TYPES
                        gif_path = make_project_activity_animation(Ts_simulation, burns, burns_BL, T, time_marker,
                                                                   animate=False, reimbursement_duration=d,
                                                                   start_year=start_year)
                    else:
                        # BURN TYPES
                        gif_path = make_project_activity_animation(Ts_simulation, burns_types, burns_types, T, time_marker,
                                                                   animate=False, reimbursement_duration=d,
                                                                   start_year=start_year,
                                                                   del_T_p_non_reimb=del_T_p_non_reimb)

                # st.image(buf)
                st.image(gif_path, width="stretch")

# --- STREAMLIT ONLY ---
if __name__ == "__main__":
    run_streamlit()