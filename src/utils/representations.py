from abc import ABC,abstractmethod
import pandas as pd

# TODO: representations as objects may not be useful

class RepresentationStrategy(ABC):
    """
    Base class for representation strategies.
    """
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def encode(self, df: pd.DataFrame) -> pd.DataFrame:
        ...

class DiscreteRepresentation(RepresentationStrategy):
    """
    Discretise based on reference range.
    Values -1/0/1 if below/within/above reference range.
    None if no value exists or no range exists or the merged range is not valid.
    """
    # TODO: implement this method
    pass

class BinaryRepresentation(RepresentationStrategy):
    """
    Binarise based on whether a value exists or not.
    0/1 for not exist/exist.
    """
    # TODO: implement this method
    pass

class ContinuousRepresentation(RepresentationStrategy):
    """
    Take the continuous value.
    """
    # TODO: implement this method
    pass

class DiscreteBinaryRepresentation(RepresentationStrategy):
    """
    Concatenate discrete and binary representation.
    """
    # TODO: implement this method
    pass

class AllRepresentation(RepresentationStrategy):
    """
    All representations (continuous, discrete, binary) concatenated.
    """
    # TODO: implement this method
    pass
