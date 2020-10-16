import math


class PerfsimModel(object):

    def __init__(self, perf_sim_params):
        self.perf_sim_params = perf_sim_params

    def simulate(self, subgraph_bin, perf_data):
        raise NotImplementedError("PerfsimModel:simulate")

    def update_perf_data(self, perf_data, instr, exec_behavior, time):
        i_stat = perf_data.execution_stats.instructions.add()
        i_stat.duration_clks = max(1, exec_behavior.total_clks())
        i_stat.start_clk = time
        i_stat.pc = instr.pc
        i_stat.instruction.CopyFrom(instr)

    def ns_to_clocks(self, ns):
        return int(
            math.ceil(float(self.perf_sim_params.common.clock_frequency) * ns / 1.e3))
