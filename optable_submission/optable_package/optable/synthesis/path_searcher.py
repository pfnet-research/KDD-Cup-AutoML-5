from optable.dataset import relation_types
import optable.synthesis.path as path_mod
import optable.dataset.relation as relation_mod


class PathSearcher(object):
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def recursion_search(self, path, depth, dataset):
        if path in self.paths:
            return
        self.paths.append(path)

        for relation in dataset.relations:
            next_depth = depth + 1
            if relation.src == relation.dst \
               and relation.type == relation_types.many_to_many:
                continue
            if relation.src == path.dst:
                self.recursion_search(path + relation, next_depth, dataset)
            if relation.dst == path.src:
                self.recursion_search(
                    path + relation.inverse, next_depth, dataset)
        return

    def search(self, dataset):
        self.paths = []
        self.recursion_search(
            path_mod.Path([], dataset.main_table_name), 0, dataset)
        with_self_mtom_paths = []
        for path in self.paths:
            for relation in dataset.relations:
                if path.dst == relation.src \
                   and relation.src == relation.dst \
                   and relation.type == relation_types.many_to_many:
                    with_self_mtom_paths.append(path + relation)
        self.paths += with_self_mtom_paths

        for path in self.paths:
            path.set_depths(dataset.depths)
        return self.paths
