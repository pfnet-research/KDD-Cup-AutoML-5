import abc


class ManipulationCandidate(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def search(self, path, dataset):
        # pathを元に、manipulationを列挙する関数
        pass


manipulation_candidates = []
manipulation_classes = []


def register_manipulation_candidate(candidate, manipulation_class):
    manipulation_candidates.append(candidate)
    manipulation_classes.append(manipulation_class)
