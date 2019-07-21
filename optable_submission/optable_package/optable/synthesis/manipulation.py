import abc


# TODO(yoshikawa): concatible groupを定義する。
class Manipulation(object, metaclass=abc.ABCMeta):
    def __init__(self):
        from optable.synthesis import learned_meta_priority
        if learned_meta_priority.USE_LEARNED_PRIORITY:
            self.__priority = \
                learned_meta_priority.calculate_meta_priority(self)
        else:
            self.__priority = self.calculate_priority()
        self.__size = self.calculate_size()

    @property
    def priority(self):
        return self.__priority

    @property
    def size(self):
        return self.__size

    @abc.abstractmethod
    def calculate_priority(self):
        pass

    @abc.abstractmethod
    def calculate_size(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def meta_feature_size():
        pass

    @abc.abstractmethod
    def meta_feature(self):
        pass

    @staticmethod
    @abc.abstractmethod
    def meta_feature_name():
        pass

    @abc.abstractmethod
    def synthesis(self):
        # synthesisする操作
        pass
