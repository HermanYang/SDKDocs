import copy

from lt_sdk.perfsim import instructions, perfsim_logging, perfsim_model
from lt_sdk.proto import lgf_pb2, sim_params_pb2


class MachineState(object):

    def __init__(self, perf_sim_params):
        self.perf_sim_params = perf_sim_params
        self.rsc_use = {}

        for e in sim_params_pb2.MosaicParams.DESCRIPTOR.enum_types:
            if e.name == "ResourceType":
                for d in e.values:
                    self.rsc_use[d.number] = 0

        # for data dependency tracking
        self.in_flight_pcs = set()
        self.complete_pcs = set()

    def __str__(self):
        return "rsc_use[{0}]".format(", ".join("{0}: {1}".format(k,
                                                                 v) for k,
                                               v in self.rsc_use.items()))

    def copy(self):
        ret = MachineState(self.perf_sim_params)
        ret.rsc_use = copy.copy(self.rsc_use)
        ret.in_flight_pcs = copy.copy(self.in_flight_pcs)
        ret.complete_pcs = copy.copy(self.complete_pcs)
        return ret

    def future(self, decoded):
        """Return a copy of this state where reqs have been freed."""
        ret = self.copy()
        ret.complete(decoded)

        return ret

    def ok(self, reqs):
        for k, v in reqs.requires.items():
            assert (k in self.perf_sim_params.mosaic.num_resources)
            if self.rsc_use[k] + v > self.perf_sim_params.mosaic.num_resources[k]:
                return False
        return True

    def allocate(self, reqs):
        for k, v in reqs.requires.items():
            self.rsc_use[k] += v

    def free(self, reqs):
        for k, v in reqs.frees.items():
            self.rsc_use[k] -= v

    def complete(self, decoded):
        self.complete_pcs.add(decoded.instr.pc)
        self.in_flight_pcs.remove(decoded.instr.pc)
        self.free(decoded.reqs)


class DecodedInstr(object):

    def __init__(self, instr):
        self.instr = instr
        which_node = instr.node.WhichOneof("node")
        if which_node not in instructions.INSTR_MODEL_FNS:
            raise ValueError("Couldn't find {0} model: {1}".format(which_node, instr))
        self.model = instructions.INSTR_MODEL_FNS[which_node](instr)
        self.reqs = self.model.requires()


class IssuedInstr(object):

    def __init__(self, dec, start_time):
        self.decoded = dec
        self.start_time = start_time
        self.exec_beh = dec.model.model()
        self.finish_time = self.start_time + self.exec_beh.total_clks()


class InstructionIssuer(object):

    def __init__(self, sub_bin, mach_state, perf_sim_params, hw_specs):
        self.hw_specs = hw_specs
        self.perf_sim_params = perf_sim_params
        self.time = 0
        self.sub_bin = sub_bin
        self.machine_state = mach_state
        self.instr_cnt = 0  # order in sub_bin.instr, not pc
        self.window = {}

        # A list of IssuedInstr sorted by finish time.
        self.in_flight = []

    def can_run(self, dec, state):
        perfsim_logging.log("Considering {0}".format(dec.instr.pc),
                            perfsim_logging.LogLevel.ISSUE_WINDOW)
        # Data dependencies
        deps = [dec.instr.pc - x for x in dec.instr.dependent_pcs_distance]
        perfsim_logging.log("...deps: {0}".format(deps),
                            perfsim_logging.LogLevel.DEPENDENCIES)
        ok = True
        for d in deps:
            if d not in state.complete_pcs:
                if d in state.in_flight_pcs:
                    in_fl = None
                    for iss in self.in_flight:
                        if iss.decoded.instr.pc == d:
                            in_fl = iss
                    ok = ((dec.model.can_pipeline()
                           and in_fl.decoded.model.can_pipeline()) or
                          (dec.model.is_opu_node() and in_fl.decoded.instr.node.HasField(
                              lgf_pb2.LNF.apw.DESCRIPTOR.name)))
                    perfsim_logging.log(
                        "... pipelined data hazard, dep {0} - {1}".format(
                            d,
                            dec.model.can_pipeline()),
                        perfsim_logging.LogLevel.HAZARDS)
                else:
                    perfsim_logging.log("... data hazard, dep {0}".format(d),
                                        perfsim_logging.LogLevel.HAZARDS)
                    ok = False
                    break

        # Resource hazards
        rsc_good = state.ok(dec.reqs)
        ok = ok and rsc_good
        perfsim_logging.log("... rsc hazard - {0}".format(not rsc_good),
                            perfsim_logging.LogLevel.HAZARDS)
        if not rsc_good:
            perfsim_logging.log("    reqs - {0}".format(dec.reqs),
                                perfsim_logging.LogLevel.HAZARDS)
            perfsim_logging.log("    state - {0}".format(state),
                                perfsim_logging.LogLevel.HAZARDS)

        return ok

    def to_run(self, state):
        for pc in sorted(self.window.keys()):
            if self.can_run(self.window[pc], state):
                return self.window[pc]
        return None

    def architectural_latency(self, dec, state):
        ret = dec.model.issue_latency(state)
        if dec.model.accesses_umem():
            ret += self.hw_specs.umem_num_banks // 2
        perfsim_logging.log("arch latency: {0}".format(ret),
                            perfsim_logging.LogLevel.ARCH_MODEL)
        return ret

    def next(self):
        # Decode up to window size
        while (len(self.window) < self.perf_sim_params.mosaic.issue_window_size
               and self.instr_cnt < len(self.sub_bin.instr)):
            i = self.sub_bin.instr[self.instr_cnt]
            self.window[i.pc] = DecodedInstr(i)
            self.instr_cnt += 1

        # Retire any completed instructions
        while self.in_flight and self.in_flight[0].finish_time < self.time:
            issued = self.in_flight[0]
            perfsim_logging.log(
                "Completing {0} at time {1}, start {2} finished at {3}".format(
                    issued.decoded.instr.pc,
                    self.time,
                    issued.start_time,
                    issued.finish_time),
                perfsim_logging.LogLevel.COMPLETE)
            self.machine_state.complete(issued.decoded)

            last_retired = self.in_flight[0]
            del self.in_flight[0]

        if not self.window:
            while self.in_flight:
                issued = self.in_flight[0]
                perfsim_logging.log(
                    "Completing {0} at time {1}, start {2} finished at {3}".format(
                        issued.decoded.instr.pc,
                        self.time,
                        issued.start_time,
                        issued.finish_time),
                    perfsim_logging.LogLevel.COMPLETE)
                self.machine_state.complete(issued.decoded)

                last_retired = self.in_flight[0]
                del self.in_flight[0]
            return None, last_retired.finish_time

        # Find the decoded instruction that can start in the closest time.
        running_state = self.machine_state
        to_run = self.to_run(running_state)
        i = 0
        to_start = self.time

        while not to_run and i < len(self.in_flight):
            perfsim_logging.log(
                "---Inflight {0} of {1} ---".format(i + 1,
                                                    len(self.in_flight)),
                perfsim_logging.LogLevel.ISSUE_WINDOW)
            running_state = running_state.future(self.in_flight[i].decoded)
            to_run = self.to_run(running_state)
            to_start = self.in_flight[i].finish_time
            i += 1

        if not to_run:
            raise RuntimeError("Could not issue an instruction")

        del self.window[to_run.instr.pc]

        issued = IssuedInstr(
            to_run,
            to_start + self.architectural_latency(to_run,
                                                  self.machine_state))
        perfsim_logging.log(
            "*****issuing {0} at time {1}: {2}".format(
                to_run.instr.pc,
                issued.start_time,
                to_run.instr.node.WhichOneof("node")),
            perfsim_logging.LogLevel.ISSUE)

        self.in_flight.append(issued)
        self.in_flight.sort(key=lambda x: x.finish_time)
        self.machine_state.in_flight_pcs.add(issued.decoded.instr.pc)

        self.time = issued.start_time
        return issued, issued.start_time


class MosaicModel(perfsim_model.PerfsimModel):

    def __init__(self, cfg, *args):
        super().__init__(cfg.sim_params.perf_params, *args)

        instructions.InstructionModel.PERF_SIM_PARAMS = cfg.sim_params.perf_params
        instructions.InstructionModel.ARCH = self

    def simulate(self, subgraph_bin, perf_data):
        ms = MachineState(self.perf_sim_params)
        ii = InstructionIssuer(subgraph_bin,
                               ms,
                               self.perf_sim_params,
                               perf_data.config.hw_specs)

        instructions.InstructionModel.PERF_SIM_PARAMS = self.perf_sim_params
        instructions.InstructionModel.HW_SPECS = perf_data.config.hw_specs

        issued, time = ii.next()

        while issued:
            ms.allocate(issued.decoded.reqs)
            self.update_perf_data(perf_data, issued.decoded.instr, issued.exec_beh, time)

            issued, time = ii.next()

        perf_data.execution_stats.total_clocks = time

    def tmem_transfer_clks(self, vecs):
        return self.ns_to_clocks(1e3 * 2 * vecs /
                                 self.perf_sim_params.mosaic.per_core_io_bandwidth)

    def hostmem_transfer_clks(self, vecs):
        return self.tmem_transfer_clks(vecs)
