import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st
from simulation import run_simulation
from project_generator import burn_curve, generate_projects
import numpy as np
import sys
import streamlit as st
from PIL import Image
import subprocess
from animate_accumulation import make_project_activity_animation

def is_debugging():
    """Checks if the current Python process is being debugged."""
    return hasattr(sys, 'gettrace') and sys.gettrace() is not None

flag_show_SD = False

# --- STREAMLIT ENTRY POINT ---
def run_streamlit():
    st.set_page_config(
        page_title="Cash Flow Simulator",
        layout="wide"
    )

    # ------------------------------------------------------------
    # GLOBAL STYLE OVERRIDES
    # ------------------------------------------------------------
    st.markdown("""
    <style>
    details > summary {
      font-size: 2.5em !important;
      font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if flag_show_SD:
        col1, col2, col3 = st.columns([1, 1, 2])  # adjust width ratio if needed
        with col1:
            with st.expander("Basic Settings", expanded=False):
                reimbursement_duration = st.slider("Reimbursement duration", 1, 10, 2)
                avg_salary = st.slider("Average Faculty Salary", 50, 200, 120, step=5)
                cash_init = st.slider("Initial Cash Balance", 0, 500, 00, step=10)
                n_projects = st.slider("Number of Projects", 2, 2000, 1500)
                max_award = st.slider("Max Award Amount", 100, 2000, 1000, step=50)
    else:
        col_spacer1, col2, col_spacer2, col3, col_spacer3 = st.columns([0.2, 2, 0.5, 3, 0.2])

    with col2:
        # Try to save the state, but slider change immediately causes rerun so del_T1_non_reimb is not saved
        #  (but seems to be saved the second time the slider is changed)
        for key, default in {
            "del_T_p_non_reimb": (0, 0.),
            "del_Tidc_rate": (0, 0.)
        }.items():
            if key not in st.session_state:
                st.session_state[key] = default

        # with st.expander("Award information", expanded=True):
        st.markdown("#### Award information")  # renders like st.subheader

        with st.expander("ℹ️ Summary of simulation procedure"):
            st.markdown("""
            A series of projects are simulated as follows:
            - The **total number** of projects is set by the *1st slider*. 
            - **Durations** are uniformly distributed from 12 to 36 months.
            - The average **burn rate** is $100,000 per month.
            - **Burn rate shapes** are set by the *dropdown menu* below.
            - The **simulation display** begins at time 0 (months), occurring after some of the projects
              have started. The simulation **end time** is set by the *4th slider* below.
            - Project **start times** are uniformly distributed starting 36 months prior to
              time 0 until the simulation end time.
            
            A decrease in award amounts is simulated as follows:
            - The *2nd slider* below sets the **month at which all awards with subsequent start times** have
              reduced award amounts.
            - The **reduced award amount**, specified as a percent of the original award amount,
              is specified by the *3rd slider* below.
            """)

        n_grants = st.slider("Number of awards", 10, 100, 30, 10,
                             help="Start times are uniformly distributed starting 36 months prior to " +
                                  "time 0 that is displayed on the charts.")
        decline_month = st.slider("Month new awards experience reduced burn rate", 0, 60, 30, 10,
                                  help="After this month, all award amounts are reduced.")
        decline_factor = st.slider("Decline as percentage of original award (%)",
                                10, 100, 50, 5,
                                help="Amount that each award is reduced.") / 100
        n_months = st.slider("Simulation duration (months)", 36, 84, 72, 12,
                                help=".")
        # show_gantt = st.checkbox("Show Gantt Chart", True)
        # seed = st.number_input("Random seed", 0, 9999, 3)
        seed = 3
        burn_shape = st.selectbox("Burn curve shape", ["bell", "constant", "early", "late"], 1,
                                help="Shape of burn rate for every award.")

        if flag_show_SD:
            with st.expander("Non-reimbursable projects", expanded=False):
                p_non_reimb = st.slider("Probability of Non-reimbursable Project", 0.0, 1.0, 0.00)
                p_delayed_NOA = st.slider("Probability of Delayed NOA", 0.0, 1.0, 0.00)
                # show_del_non_reimb = st.toggle("Change non-reimbursable rate mid-year")
                # if show_del_non_reimb:
                del_T1_non_reimb = st.slider("Time of change in non reimbursable projects", 0, 50,
                                          st.session_state.del_T_p_non_reimb[0])
                del_p_non_reimb = st.slider("Increase in probability of non reimbursable", 0.0, 1.0,
                                          st.session_state.del_T_p_non_reimb[1])
                del_T_p_non_reimb = (del_T1_non_reimb, del_p_non_reimb)
                # else:
                #     del_T_p_non_reimb = st.session_state.del_T_p_non_reimb

            with st.expander("IDC", expanded=False):
                idc_rate = st.slider("Indirect Cost Rate (%)", 0.0, 1.0, 0.55)
                # show_del_idc_rate = st.toggle("Decrease idc rate mid-year")
                # if show_del_idc_rate:
                del_T1_idc_rate = st.slider("Time of decrease in idc rate", 0, 50, 0)
                del_p_idc_rate = st.slider("Decrease in idc rate", 0.0, 1.0, 0.0)
                del_T_p_idc_rate = (del_T1_idc_rate, del_p_idc_rate)
                # else:
                #     del_T_p_idc_rate = (-1, 0.)

    with col3:
        if flag_show_SD:
            (cash_history, non_reimb1, non_reimb2, idc_log, inst_paid_log, total_avg_reimbursement,
             total_spend_reimbursable, total_spend_non_reimbursable) = run_simulation(
                avg_salary, cash_init, p_non_reimb, n_projects, max_award, idc_rate, del_T_p_non_reimb,
                del_T_p_idc_rate,
                reimbursement_duration=reimbursement_duration, p_delayed_NOA=p_delayed_NOA
            )

            fig, ax = plt.subplots(1, 1, figsize=(9,8))

            st.subheader("Cash and Receivable Balances Over Time")
            ax.plot(cash_history, label="Cash Balance", linewidth=2)
            ax.plot(non_reimb1, label="Receivable (Reimbursable Projects)", linestyle="--")
            ax.plot(non_reimb2, label="Receivable (Non-reimbursable Projects)", linestyle=":")
            ax.set_xlabel("Time (weeks)")
            ax.set_ylabel("Amount ($MM)")
            # Format Y-axis ticks with commas
            # ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
            def thousands(x, pos):
                return f"{x / 1000:.2f}"
            ax.yaxis.set_major_formatter(FuncFormatter(thousands))
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        # Generate data
        t = np.arange(n_months)
        burns,burns_BL = generate_projects(t, n_grants, decline_month, decline_factor, n_months,
                                           seed=seed, shape=burn_shape)
        total = burns.sum(axis=0)
        total_BL = burns_BL.sum(axis=0)

        # Build the figure
        if is_debugging():
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8),
                                           gridspec_kw={'height_ratios': [2, 1]},
                                           constrained_layout=True)
            # Ribbon chart
            ax1.stackplot(t, burns, alpha=0.7)
            ax1.plot(t, total_BL, "k--", lw=2.5, label="Baseline (no decline)")
            ax1.plot(t, total, color="red", lw=2.5, label="With decline")
            ax1.axvline(decline_month, color="red", ls=":", label="Decline starts")
            ax1.set_ylabel("Monthly Spending ($M)")
            ax1.set_title("Ribbon Chart: Total Revenue Over Time")
            ax1.legend(fontsize=8, loc="upper right")
        # Gantt chart (optional)
        #     if show_gantt and ax2 is not None:
            if ax2 is not None:
                norm_burns = burns / np.max(burns, axis=1, keepdims=True)
                im = ax2.imshow(norm_burns, aspect="auto", cmap="YlOrRd",
                                extent=[t[0], t[-1], 0, burns.shape[0]])
                ax2.axvline(decline_month, color="gray", ls=":")
                ax2.set_xlabel("Month")
                ax2.set_ylabel("Project Index")
                ax2.set_title("Gantt Chart: Project Activity")
                fig.colorbar(im, ax=ax2, fraction=0.02, pad=0.02,
                             label="Relative burn intensity")
                plt.show()
        # else:
        #     fig, ax1 = plt.subplots(figsize=(9, 5))
        #     ax2 = None

        # st.markdown("""
        # **Interpretation**
        #
        # - The **left columns** let you control model parameters interactively.
        # - The **right column** updates the ribbon (and optionally the Gantt) plot in real time.
        # """)

        png_path = make_project_activity_animation(t, burns, burns_BL, n_months, decline_month,
                                                   animate=False)  # DEBUG

        st.image(png_path, width="stretch")

        if is_debugging():
            print("DEBUG MODE")
            gif_path = make_project_activity_animation(t, burns, burns_BL, n_months, decline_month) # DEBUG
        else:
            # ANIMATION
            # if st.button("Generate Animation"):
            with st.spinner("Creating animation... please wait ⏳"):
                gif_path = make_project_activity_animation(t, burns, burns_BL, n_months, decline_month)
            # st.success("Animation complete!")
            # fig, ax = plt.subplots()
            # st.pyplot(fig)

            st.image(gif_path, width="stretch")

    # st.session_state.del_T_p_non_reimb = del_T_p_non_reimb

    # st.subheader("Indirect Costs and Institution-Funded Compensation")
    # st.metric("Total Reimbursable Spending", f"${total_spend_reimbursable:,.2f}")
    # st.metric("Total Non-reimbursable Spending", f"${total_spend_non_reimbursable:,.2f}")
    # st.write(f"Total Indirect Cost Collected: ${sum(idc_log):,.2f}")
    # st.write(f"Total Institution-Paid Compensation: ${sum(inst_paid_log):,.2f}")
    # st.write(f"Total Average Reimbursement: ${total_avg_reimbursement:,.2f}")

# --- STREAMLIT ONLY ---
if __name__ == "__main__":
    run_streamlit()