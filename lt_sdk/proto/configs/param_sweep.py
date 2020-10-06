import copy


class ParamSweep(object):

    def __init__(self, name, base_fn, *args, **kwargs):
        self.name = name
        self.base_fn = base_fn
        self.args = args  # default args for base_fn, not swept over
        self.kwargs = kwargs  # string -> list

    def generate(self):
        # flatten params
        combos = [{}]
        for k, vlist in self.kwargs.items():
            new_combos = []
            for val in vlist:
                for c in combos:
                    new_c = copy.deepcopy(c)
                    new_c[k] = val
                    new_combos.append(new_c)
            combos = new_combos

        for c in combos:
            new_obj = self.base_fn(*self.args, **c)
            cfg_name = []
            for k, v in c.items():
                cfg_name.append("{0}^^{1}".format(k, str(v)))
            yield new_obj, "~~".join(cfg_name)
