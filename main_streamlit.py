import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import streamlit as st
from simulation import run_simulation

# --- STREAMLIT ENTRY POINT ---
def run_streamlit():
    st.set_page_config(
        page_title="Cash Flow Simulator",
        layout="wide"
    )
    col1, col2, col3 = st.columns([1, 1, 2])  # adjust width ratio if needed
    with col1:
        with st.expander("Basic Settings", expanded=True):
            avg_salary = st.slider("Average Faculty Salary", 50, 200, 120, step=5)
            cash_init = st.slider("Initial Cash Balance", 0, 500, 00, step=10)
            n_projects = st.slider("Number of Projects", 1000, 2000, 1500)
            max_award = st.slider("Max Award Amount", 100, 2000, 1000, step=50)

    with col2:
        # Try to save the state, but slider change immediately causes rerun so del_T1_non_reimb is not saved
        #  (but seems to be saved the second time the slider is changed)
        for key, default in {
            "del_T_p_non_reimb": (-1, 0.),
            "del_Tidc_rate": (-1, 0.)
        }.items():
            if key not in st.session_state:
                st.session_state[key] = default

        with st.expander("Non-reimbursable projects", expanded=True):
            p_non_reimb = st.slider("Probability of Non-reimbursable Project", 0.0, 1.0, 0.00)
            show_del_non_reimb = st.toggle("Change non-reimbursable rate mid-year")
            if show_del_non_reimb:
                del_T1_non_reimb = st.slider("Time of change in non reimbursable projects", 0, 50,
                                          st.session_state.del_T_p_non_reimb[0]+1)
                del_p_non_reimb = st.slider("Increase in probability of non reimbursable", 0.0, 1.0,
                                          st.session_state.del_T_p_non_reimb[1])
                del_T_p_non_reimb = (del_T1_non_reimb-1, del_p_non_reimb)
            else:
                del_T_p_non_reimb = st.session_state.del_T_p_non_reimb

        with st.expander("IDC", expanded=True):
            idc_rate = st.slider("Indirect Cost Rate (%)", 0.0, 1.0, 0.55)
            show_del_idc_rate = st.toggle("Decrease idc rate mid-year")
            if show_del_idc_rate:
                del_T1_idc_rate = st.slider("Time of decrease in idc rate", 0, 50, 0)
                del_p_idc_rate = st.slider("Decrease in idc rate", 0.0, 1.0, 0.0)
                del_T_p_idc_rate = (del_T1_idc_rate-1, del_p_idc_rate)
            else:
                del_T_p_idc_rate = (-1, 0.)

    (cash_history, non_reimb1, non_reimb2, idc_log, inst_paid_log, total_avg_reimbursement,
        total_spend_reimbursable, total_spend_non_reimbursable) = run_simulation(
        avg_salary, cash_init, p_non_reimb, n_projects, max_award, idc_rate, del_T_p_non_reimb, del_T_p_idc_rate
    )

    with col3:
        st.subheader("Cash and Receivable Balances Over Time")
        fig, ax = plt.subplots()
        ax.plot(cash_history, label="Cash Balance", linewidth=2)
        ax.plot(non_reimb1, label="Receivable (Reimbursable Projects)", linestyle="--")
        ax.plot(non_reimb2, label="Receivable (Non-reimbursable Projects)", linestyle=":")
        ax.set_xlabel("Time (weeks)")
        ax.set_ylabel("Amount ($MM)")
        # Format Y-axis ticks with commas
        # ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))
        def thousands(x, pos):
            return f"{x / 1000:.0f}"
        ax.yaxis.set_major_formatter(FuncFormatter(thousands))
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

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