import random
from stocks_new import Personnel, Project, CashStock

def run_simulation(avg_salary, cash_init, p_non_reimb, n_projects, max_award, idc_rate,
                   del_T_p_non_reimb=(-1,0.), del_T_p_idc_rate=(-1,0.), T=152, reimbursement_duration=2,
                   p_delayed_NOA=0):
    # --- Setup ---
    projects = []
    for i in range(n_projects):
        is_non_reimbursable = random.random() < p_non_reimb
        if is_non_reimbursable:
            reimbursement_duration = -1 # overwrite
        else:
            if random.random() < p_delayed_NOA:
                proj_reimbursement_duration = random.randint(reimbursement_duration,52)
            else:
                proj_reimbursement_duration = reimbursement_duration
        duration = random.randint(51, 52) if reimbursement_duration > 0 else 52  # avoid very short projects
        start = random.randint(-51, 50)  # project can start before simulation
        end = start + duration
        if end < 0 or start >= T:
            continue  # completely outside simulation window

        proj = Project(
            f"Project {i+1}", start, end,
            award_amount = random.randint(int(0.75 * max_award), max_award),
            reimbursement_duration = proj_reimbursement_duration
        )
        projects.append(proj)

    total_spend_reimbursable = 0.0
    total_spend_non_reimbursable = 0.0

    for t in range(T):
        spend_this_step_reimb = 0.0
        spend_this_step_non_reimb = 0.0

        for proj in projects:
            rate = round(proj.spend_rate(t, False)) # don't update non-reimbursable
            if proj.reimbursement_duration == -1:
                spend_this_step_non_reimb += rate
            else:
                spend_this_step_reimb += rate

        total_spend_reimbursable += spend_this_step_reimb
        total_spend_non_reimbursable += spend_this_step_non_reimb

    total_faculty_salary = total_spend_non_reimbursable + round(total_spend_reimbursable*(1+idc_rate))
    project_faculty_salary = total_spend_non_reimbursable + total_spend_reimbursable
    alpha = project_faculty_salary / total_faculty_salary
    N = int(total_faculty_salary / avg_salary)
    # alpha might not be needed
    faculty = Personnel("faculty", N=N, average_salary=avg_salary, alpha=alpha)

    cash = CashStock("Main Cash", initial_value=cash_init)
    for proj in projects:
        cash.add_outflow(proj.spend_rate)

    cash_history = []
    receivable_reimbursable = []
    receivable_non_reimbursable = []
    idc_log = []
    inst_paid_log = []
    reimbursement_by_project = [[] for _ in projects]

    for t in range(T):
        # cash.update(t) # calls spend rate
        faculty_inst_pay = faculty.total_comp()

        if t==int(del_T_p_non_reimb[0]): # time of possible change
            for proj in projects: # for projects with reimbursement that haven't started
                if proj.reimbursement_duration!=-1 and \
                    proj.start_date > t and \
                    random.random() < del_T_p_non_reimb[1] / (1 - p_non_reimb):
                        proj.reimbursement_schedule = set() # non-reimbursable project
                        proj.reimbursement_duration = -1

        if t==int(del_T_p_idc_rate[0]): # time of possible change
            idc_rate -= del_T_p_idc_rate[1]

        # IDC and reimbursements
        for i, proj in enumerate(projects):
            proj_spend = proj.spend_rate(t)
            proj_reimb = proj.reimbursement_rate(t)
            reimbursement_by_project[i].append(proj_reimb)
            faculty_inst_pay -= proj_spend
            cash.value -= proj_spend

            if proj_reimb > 0:
                idc = round(proj_reimb * idc_rate)
                cash.value += proj_reimb + idc
                cash.reimbursements.append((t, proj_reimb))
                idc_log.append(idc)

            proj.update_receivable(t)  # reset receivable to 0

        # Track unreimbursed and institution-paid comp
        receivable_reimbursable.append(sum(
            p.receivable_amount for p in projects if p.reimbursement_duration != -1
        ))
        receivable_non_reimbursable.append(sum(
            p.receivable_amount for p in projects if p.reimbursement_duration == -1
        ))
        # inst_paid_total = faculty.institution_paid()
        if t>0:
            cash.value -= faculty_inst_pay
            inst_paid_log.append(faculty_inst_pay)
        else:
            inst_paid_log.append(0)

        cash_history.append(cash.value)

    avg_reimb_per_project = [round(sum(r) / T) for r in reimbursement_by_project]
    total_avg_reimbursement = round(sum(avg_reimb_per_project))

    return (cash_history, receivable_reimbursable, receivable_non_reimbursable, idc_log, inst_paid_log,
            total_avg_reimbursement, total_spend_reimbursable, total_spend_non_reimbursable)