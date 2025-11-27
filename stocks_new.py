import json

import numpy as np


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
    def __init__(self, name, start_date, end_date, award_amount, reimbursement_duration=2,
                 change_in_duration_date=10**10, reimbursement_duration_2=2, t=0):
        # t is the start time of the simulation
        self.name = name
        self.start_date = round(start_date)
        self.end_date = round(end_date)
        self.award_amount = award_amount
        self.duration = round(end_date - start_date)
        self.reimbursement_duration = round(reimbursement_duration)
        self.change_in_duration_date = round(change_in_duration_date)
        self.reimbursement_duration_2 = round(reimbursement_duration_2)
        self.NOA_delay = 0.0
        self.print = False # v2
        self.type = 0

        self.receivable_amount = 0.0
        self.reimbursement_schedule = set()

        if reimbursement_duration > 0:
            self.set_reimbursement_schedule(start_date, end_date, reimbursement_duration,
                                            change_in_duration_date, reimbursement_duration_2)
        if self.start_date < t < self.end_date:
            remaining_duration = self.end_date - t
            self.award_balance = award_amount * remaining_duration / self.duration # scale amount by remaining after t
        else:
            self.award_balance = award_amount

        self.print_json('__init__')

    def set_reimbursement_schedule(self, start_date, end_date, reimbursement_duration,
                                   change_in_duration_date, reimbursement_duration_2):
        self.reimbursement_schedule = set()
        t = round(start_date)
        while t < round(min(change_in_duration_date, end_date)):
            reimbursement_duration = round(reimbursement_duration)
            if reimbursement_duration >= 2:
                rand_duration = np.random.randint(reimbursement_duration - 1)
            else:
                rand_duration = 0
            t += reimbursement_duration - rand_duration
            self.reimbursement_schedule.add(t)
        if change_in_duration_date < end_date:
            t2 = round(start_date)
            while t2 < round(end_date):
                if reimbursement_duration_2 >= 2:
                    rand_duration = np.random.randint(reimbursement_duration_2 - 1)
                else:
                    rand_duration = 0
                t2 += reimbursement_duration_2 - rand_duration
                if t2 > change_in_duration_date and t2 >= t + reimbursement_duration:
                    self.reimbursement_schedule.add(t2)
            # self.reimbursement_schedule.add(t2 + reimbursement_duration_2)
        # else:
            # self.reimbursement_schedule.add(t + reimbursement_duration)

        # while t < round(end_date):
        #     if t > round(change_in_duration_date):
        #         reimbursement_duration = reimbursement_duration_2
        #     reimbursement_duration = round(reimbursement_duration)
        #     t += reimbursement_duration
        #     self.reimbursement_schedule.add(t)
        # self.reimbursement_schedule.add(t + reimbursement_duration)

    def print_json(self, heading, key=None):
        if self.print:
            print(heading)
            if key is None:
                # print the whole object
                print(json.dumps(self.__dict__, indent=4, sort_keys=True, default=str))
            else:
                # print just one key–value pair
                value = self.__dict__.get(key, "<missing>")
                print(json.dumps({key: value}, indent=4, sort_keys=True, default=str))

    def spend_rate(self, t, flag_record_receivable=True, reduced_burn_rate=100.):
        if self.start_date < t < self.end_date:
            remaining_duration = self.end_date - t
            if reduced_burn_rate < 100.:
                rate = round(self.award_amount / self.duration) * reduced_burn_rate
            else:
                rate = (self.award_balance / remaining_duration)
            # if recording receivables, add the burn rate to the receivable amount
            if flag_record_receivable:
                self.print_json('spend_rate (before change)', key='receivable_amount')
                self.receivable_amount += rate
                self.award_balance -= rate
                self.print_json('spend_rate (after change)', key='receivable_amount')
            return rate
        return 0.0

    def reimbursement_rate(self, t):
        # if t is in the reimbursement schedule, return the accumulated receivable amount, otherwise, 0
        if t in self.reimbursement_schedule: # reimbursement_schedule = set() if reimbursement_duration = -1
            self.print_json('reimbursement_rate', key='receivable_amount')
            amt = self.receivable_amount
            # self.receivable_amount = 0.0
            return amt
        return 0.0

    def update_receivable(self, t):
        if t in self.reimbursement_schedule:
            self.print_json('update_receivable', key='receivable_amount')
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
