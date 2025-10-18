import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st
from simulation import run_simulation
from project_generator import burn_curve, generate_projects
import numpy as np
import streamlit as st
from PIL import Image
import subprocess
from animate_accumulation import make_project_activity_animation

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

    col1, col2, col3 = st.columns([1, 1, 2])  # adjust width ratio if needed
    with col1:
        with st.expander("Basic Settings", expanded=False):
            reimbursement_duration = st.slider("Reimbursement duration", 1, 10, 2)
            avg_salary = st.slider("Average Faculty Salary", 50, 200, 120, step=5)
            cash_init = st.slider("Initial Cash Balance", 0, 500, 00, step=10)
            n_projects = st.slider("Number of Projects", 2, 2000, 1500)
            max_award = st.slider("Max Award Amount", 100, 2000, 1000, step=50)

    with col2:
        # Try to save the state, but slider change immediately causes rerun so del_T1_non_reimb is not saved
        #  (but seems to be saved the second time the slider is changed)
        for key, default in {
            "del_T_p_non_reimb": (0, 0.),
            "del_Tidc_rate": (0, 0.)
        }.items():
            if key not in st.session_state:
                st.session_state[key] = default

        with st.expander("Project Overlap", expanded=True):
            st.markdown("#### Project Parameters")  # renders like st.subheader
            n_grants_baseline = st.slider("Grants before decline", 10, 30, 20)
            decline_month = st.slider("Month new awards decline starts", 0, 60, 60)
            decline_factor = st.slider("Remaining rate after decline (%)", 10, 100, 70) / 100
            n_months = st.slider("Simulation months", 36, 84, 60)
            # show_gantt = st.checkbox("Show Gantt Chart", True)
            # seed = st.number_input("Random seed", 0, 9999, 3)
            seed = 3
            burn_shape = st.selectbox("Burn curve shape", ["bell", "early", "late"])

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

    (cash_history, non_reimb1, non_reimb2, idc_log, inst_paid_log, total_avg_reimbursement,
        total_spend_reimbursable, total_spend_non_reimbursable) = run_simulation(
        avg_salary, cash_init, p_non_reimb, n_projects, max_award, idc_rate, del_T_p_non_reimb, del_T_p_idc_rate,
        reimbursement_duration=reimbursement_duration, p_delayed_NOA=p_delayed_NOA
    )

    with col3:
        flag_show_SD = False
        if flag_show_SD:
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
        burns = generate_projects(t, n_grants_baseline, decline_month, decline_factor, n_months,
                                  seed=seed, shape=burn_shape)
        total = burns.sum(axis=0)
        baseline = generate_projects(t, n_grants_baseline, 0, 1.0, n_months,
                                     seed=seed, shape=burn_shape)
        total_baseline = baseline.sum(axis=0)

        # Build the figure
        # if show_gantt:
        #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8),
        #                                    gridspec_kw={'height_ratios': [2, 1]},
        #                                    constrained_layout=True)
        # else:
        #     fig, ax1 = plt.subplots(figsize=(9, 5))
        #     ax2 = None
        #
        # # Ribbon chart
        # ax1.stackplot(t, burns, alpha=0.7)
        # ax1.plot(t, total_baseline, "k--", lw=2.5, label="Baseline (no decline)")
        # ax1.plot(t, total, color="red", lw=2.5, label="With decline")
        # ax1.axvline(decline_month, color="gray", ls=":", label="Decline starts")
        # ax1.set_ylabel("Monthly Spending ($M)")
        # ax1.set_title("Ribbon Chart: Total Revenue Over Time")
        # ax1.legend(fontsize=8, loc="upper right")

        # Gantt chart (optional)
        # if show_gantt and ax2 is not None:
        #     norm_burns = burns / np.max(burns, axis=1, keepdims=True)
        #     im = ax2.imshow(norm_burns, aspect="auto", cmap="YlOrRd",
        #                     extent=[t[0], t[-1], 0, burns.shape[0]])
        #     ax2.axvline(decline_month, color="gray", ls=":")
        #     ax2.set_xlabel("Month")
        #     ax2.set_ylabel("Project Index")
        #     ax2.set_title("Gantt Chart: Project Activity")
        #     fig.colorbar(im, ax=ax2, fraction=0.02, pad=0.02,
        #                  label="Relative burn intensity")

        st.markdown("""
        **Interpretation**
    
        - The **left columns** let you control model parameters interactively.
        - The **right column** updates the ribbon (and optionally the Gantt) plot in real time.
        """)

        t = np.arange(n_months)
        burns = generate_projects(t, n_grants_baseline, decline_month, decline_factor,
                                  n_months, seed=seed, shape=burn_shape)

        # ANIMATION
        if st.button("Generate Animation"):
            with st.spinner("Creating animation... please wait ⏳"):
                gif_path = make_project_activity_animation(t, burns, n_months)
            st.success("Animation complete!")
            fig, ax = plt.subplots()
            # st.pyplot(fig)

            st.image(gif_path, caption="Gantt + Animated Stacked Area", use_container_width=True)

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