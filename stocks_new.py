
class Personnel:
    def __init__(self, role, N, average_salary, alpha):
        self.role = role
        self.N = N
        self.average_salary = average_salary
        self.alpha = alpha

    def total_comp(self):
        return round(self.N * self.average_salary / 52.0)

    def project_paid(self):
        return round(self.alpha * self.total_comp())

    def institution_paid(self):
        return round((1 - self.alpha) * self.total_comp())


class Project:
    def __init__(self, name, start_date, end_date, award_amount, reimbursement_duration=2):
        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self.award_amount = award_amount
        self.duration = end_date - start_date
        self.reimbursement_duration = reimbursement_duration
        self.award_balance = award_amount

        self.receivable_amount = 0.0
        self.reimbursement_schedule = set()

        if reimbursement_duration != -1:
            t = start_date + reimbursement_duration
            while t < end_date:
                self.reimbursement_schedule.add(t)

                t += reimbursement_duration
            self.reimbursement_schedule.add(end_date)

    def spend_rate(self, t, flag_record_receivable=True):
        if self.start_date < t < self.end_date:
            rate = round(self.award_amount / self.duration)
            if flag_record_receivable:
                self.receivable_amount += rate
            return rate
        return 0.0

    def reimbursement_rate(self, t):
        if t in self.reimbursement_schedule: # reimbursement_schedule = () if reimbursement_duration = -1
            amt = self.receivable_amount
            # self.receivable_amount = 0.0
            return amt
        return 0.0

    def update_receivable(self, t):
        if t in self.reimbursement_schedule:
            self.receivable_amount = 0.0


class CashStock:
    def __init__(self, name, initial_value=0.0):
        self.name = name
        self.value = initial_value
        self.inflows = []
        self.outflows = []
        self.history = []
        self.reimbursements = []

    def add_inflow(self, flow_in):
        self.inflows.append(flow_in)

    def add_outflow(self, flow_out):
        self.outflows.append(flow_out)

    def compute_net_flow(self, t):
        total_in = sum(fn(t) if callable(fn) else fn for fn in self.inflows)
        total_out = sum(fn(t) if callable(fn) else fn for fn in self.outflows)
        return round(total_in - total_out)

    def update(self, t):
        net_flow = self.compute_net_flow(t)
        self.value += net_flow
        self.history.append((t, self.value))
        return self.value
