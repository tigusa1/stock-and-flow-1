import random
import numpy as np
from stocks_new import Personnel, Project, CashStock

# === MODIFICATIONS ===
# run_simulation()
#   return np.array(grants),np.array(grants_BL)
#   grants format follows generate_projects()
#     grants = []
#     grants.append(burn_curve(t, start, 40, peak, shape))  # DEBUG [ 0., 0., 1., 1., ..., 1., 0. ... ]
#   add decline_month, decline_factor

def run_simulation(cash_init, p_non_reimb, n_projects, max_award, idc_rate,
                   future_T_reimbursement_delay,
                   del_T_p_non_reimb, del_T_p_idc_rate, p_T1_delayed_NOA, reduction_T1_burn_rate,
                   T=52, reimbursement_duration=2,
                   P_start=-51, P_end=52, P_duration=52):

    # Initialize projects including project status (reimbursable or not)
    #   and reimbursement duration
    # print(f"p_non_reimb: {p_non_reimb:.4f}, "
    #       f"del_T_p_non_reimb: {del_T_p_non_reimb[0]:.0f}, {del_T_p_non_reimb[1]:.4f}")
    print(f"future_T_reimbursement_delay: "
          f"{future_T_reimbursement_delay[0]},  {future_T_reimbursement_delay[1]}")
    projects = []
    proj_types = np.zeros(n_projects)
    start = P_start
    i_print = 0
    for i in range(n_projects):
        # start = random.randint(P_start, P_end+1)  # project can start before simulation
        start = start + 1 if start < P_end+1 else P_start  # v2, sequential start time modulo P_end
        end = start + P_duration
        # if end < 0 or start >= T:
        #     continue  # completely outside simulation window

        # DETERMINE IF THE PROJECT IS NON-REIMBURSABLE
        if start < del_T_p_non_reimb[0]:
            prob_non_reimb = p_non_reimb
        else:
            prob_non_reimb = p_non_reimb + del_T_p_non_reimb[1]

        if random.random() < prob_non_reimb:
            proj_reimbursement_duration = -1 # overwrite
            # print(f"Project {i+1} is non-reimbursable.")
        else:
            proj_reimbursement_duration = reimbursement_duration

        proj = Project(
            f"Project {i+1}", start, end, max_award,
            reimbursement_duration = proj_reimbursement_duration, # set to -1 if no reimbursement
            change_in_duration_date = future_T_reimbursement_delay[0],
            reimbursement_duration_2 = future_T_reimbursement_delay[1]
        )

        if proj_reimbursement_duration == -1:
            proj.type = 1
            proj.NOA_delay = 10**10

        # CREATE DELAY IN NOA
        if random.random() < p_T1_delayed_NOA[1] and p_T1_delayed_NOA[0] > 0 and \
                proj_reimbursement_duration > 0:
            #  keep only scheduled dates after delay
            proj.set_reimbursement_schedule(
                proj.start_date + p_T1_delayed_NOA[0] + proj_reimbursement_duration,
                proj.end_date, proj_reimbursement_duration,
                future_T_reimbursement_delay[0], future_T_reimbursement_delay[1])
            proj.NOA_delay = p_T1_delayed_NOA[0] # amount of delay
            proj.type = 2

        # === PRINT PROJECT DICTIONARY ===
        # if proj.start_date > future_T_reimbursement_delay[0] - future_T_reimbursement_delay[1]*2 and \
        #         i_print < 9:
        #     print(proj.__dict__)
        #     i_print += 1

        projects.append(proj)

    for i, proj in enumerate(projects):
        proj_types[i] = proj.type

    # GET TOTAL SPENDING
    total_spend_reimbursable = 0.0
    total_spend_non_reimbursable = 0.0

    for t in range(T):
        spend_this_step_reimb = 0.0
        spend_this_step_non_reimb = 0.0

        for proj in projects:
            # only computing spend rate so don't update receivable amount
            rate = (proj.spend_rate(t, False))
            if proj.reimbursement_duration == -1:
                spend_this_step_non_reimb += rate
            else:
                spend_this_step_reimb += rate

        total_spend_reimbursable += spend_this_step_reimb
        total_spend_non_reimbursable += spend_this_step_non_reimb

    # UPDATE cash STOCK
    cash = CashStock("Main Cash", initial_value=cash_init)
    for proj in projects:
        cash.add_outflow(proj.spend_rate)

    # SIMULATE SPENDING AND REIMBURSEMENTS AT EACH TIME
    cash_history = []
    receivable_reimbursable = []
    receivable_non_reimbursable = []
    idc_log = [] # log of amount idc paid
    idc_2_log = [] # log of reduced idc paid
    inst_paid_log = [] # log of amount the institution paid to faculty
    reimbursement_by_project = [[] for _ in projects]
    spend_by_project = [[] for _ in projects]

    np_grants  = np.zeros((n_projects,T)) # v2
    np_grants_BL = np.zeros((n_projects,T)) # v2

    for t in range(P_start,T): # need to include all projects
        # cash.update(t) # calls spend rate
        total_spend = 0.0
        faculty_inst_pay = 0.0

        # CHANGE IN IDC
        if t>=del_T_p_idc_rate[0]: # time of possible change
            idc_rate_2 = del_T_p_idc_rate[1]
        else:
            idc_rate_2 = idc_rate

        # IDC and reimbursements
        idc = 0.0
        idc_2 = 0.0

        for i, proj in enumerate(projects):
            # DELAYED NOA, BEFORE END OF DELAY, AFTER REDUCED BURN RATE TIME START
            if proj.NOA_delay > 0 and t < proj.start_date + proj.NOA_delay and \
                    t > proj.start_date + reduction_T1_burn_rate[0]:
                proj_spend = proj.spend_rate(t, reduced_burn_rate=reduction_T1_burn_rate[1])
            else:
                proj_spend = proj.spend_rate(t)

            proj_reimb = proj.reimbursement_rate(t) # if t in self.reimbursement_schedule

            if t>=0:
                reimbursement_by_project[i].append(proj_reimb)
                spend_by_project[i].append(proj_spend)
                cash.value -= proj_spend
                np_grants[i,t] = proj_spend # v2 NEED TO CHANGE TO ~BL
                np_grants_BL[i,t] = proj_spend # v2 NEED TO CHANGE TO BL

                if proj_reimb > 0:
                    # if positive reimbursement, add reimbursement and the idc to the cash stock and save in
                    #   reimbursement record and idc log
                    idc += round(proj_reimb * idc_rate)
                    idc_2 += round(proj_reimb * idc_rate_2)
                    cash.value += proj_reimb + idc
                    cash.reimbursements.append((t, proj_reimb))

            proj.update_receivable(t)  # reset receivable to 0 because the receivable was reimbursed

        if t>=0:
            # Track unreimbursed and institution-paid comp
            idc_log.append(idc)
            idc_2_log.append(idc_2)

            receivable_reimbursable.append(sum(
                p.receivable_amount for p in projects if p.reimbursement_duration != -1
            ))
            receivable_non_reimbursable.append(sum(
                p.receivable_amount for p in projects if p.reimbursement_duration == -1
            ))

            # inst_paid_log
            cash.value -= faculty_inst_pay
            inst_paid_log.append(faculty_inst_pay)

            cash_history.append(cash.value)

    avg_reimb_per_project = [round(sum(r) / T) for r in reimbursement_by_project]
    total_avg_reimbursement = round(sum(avg_reimb_per_project))

    return (cash_history, receivable_reimbursable, receivable_non_reimbursable, idc_log, idc_2_log,
            inst_paid_log,
            total_avg_reimbursement, total_spend_reimbursable, total_spend_non_reimbursable,
            np_grants, np_grants_BL, projects, cash,
            np.array(spend_by_project), np.array(reimbursement_by_project),
            proj_types) # v2 return np grants arrays