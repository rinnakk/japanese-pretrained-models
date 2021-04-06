import collections

import numpy as np


class StatisticsReporter:
    def __init__(self):
        self.statistics = collections.defaultdict(list)

    def update_data(self, d):
        for k, v in d.items():
            if isinstance(v, (int, float)):
                self.statistics[k] += [v]

    def clear(self):
        self.statistics = collections.defaultdict(list)

    def to_string(self):
        string_list = []
        for k, v in sorted(list(self.statistics.items()), key=lambda x: x[0]):
            mean = np.mean(v)
            string_list.append("{}: {:.5g}".format(k, mean))
        return ", ".join(string_list)

    def get_value(self, key):
        if key in self.statistics:
            value = np.mean(self.statistics[key])
            return value
        else:
            return None

    def items(self):
        for k, v in self.statistics.items():
            yield k, v
