import time
import traceback

import numpy as np
from concurrent import futures

from optable.synthesis import path_searcher as path_searcher_mod
import optable.synthesis.manipulation_candidate as mc_mod
from optable import _core


class Synthesizer(object):
    def __init__(self, dataset, timer, priority_perturbation=0):
        self.priority_perturbation = priority_perturbation
        self.timer = timer
        max_depth = max([3, dataset.max_depth + 1])
        path_searcher = path_searcher_mod.PathSearcher(max_depth)
        self.__paths = path_searcher.search(dataset)

        manipulations = []
        for path in self.__paths:
            for manipulation_candidate in mc_mod.manipulation_candidates:
                manipulations += manipulation_candidate.search(
                    path, dataset)

        self.__manipulations = manipulations
        self.sort()

        self.__feature_num = None
        self.__timeout = None
        self.__start_time = None
        self.__dataset = dataset

    @property
    def manipulations(self):
        return self.__manipulations

    @property
    def paths(self):
        return self.__paths

    def sort(self):
        priorities = {manip: manip.priority
                      + self.priority_perturbation * np.random.uniform()
                      for manip in self.__manipulations}
        self.__manipulations = \
            [k for k, v in sorted(priorities.items(), key=lambda x: x[1])]

    def synthesis_at(self, index):
        if (time.time() - self.__start_time) > self.__timeout:
            return
        if self.timer.memory_usage > 13:
            return
        if self.__dataset.tables['main'].new_data_size >= self.__feature_num:
            return
        manipulation = self.__manipulations[index]
        try:
            manipulation.synthesis()
        except Exception as e:
            traceback.print_exc()
        if index % 10 == 0:
            self.timer.print("{} synthesis finished!".format(index))
            self.timer.print_memory_usage()
        if index % 100 == 0:
            _core.malloc_trim(0)
            gc.collect()

    def synthesis(self, feature_num, timeout):
        self.__feature_num = feature_num
        self.__timeout = timeout
        self.__start_time = time.time()

        self.timer.print("{} synthesis start".format(feature_num))
        with futures.ThreadPoolExecutor(max_workers=4) as executor:
            data = executor.map(
                self.synthesis_at, list(range(len(self.__manipulations))))

        if (time.time() - self.__start_time) > self.__timeout:
            self.timer.print("timeout")

        self.__feature_num = None
        self.__timeout = None
        self.__start_time = None
