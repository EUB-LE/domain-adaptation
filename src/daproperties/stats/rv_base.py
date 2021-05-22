from __future__ import annotations
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from daproperties.measures import *
import numpy as np

class rv_base(ABC):

    # Insert new divergence measures here
    # TODO: Automatically detect divergence measures.
    DIVERGENCE_MEASURES = {
        "jsd": jsd,
        "kld": kld
    }
    
    @abstractmethod
    def score_samples(self) -> np.ndarray:
        pass

    @abstractmethod
    def divergence(self, rv:rv_base, div_type:str="jsd") -> float:
        pass

    def divergence_from_distribution(self, pd_x:np.ndarray, pd_y:np.ndarray, div_type:str="jsd") -> float: 
        try:
            divergence_measure = self.DIVERGENCE_MEASURES[div_type]
        except KeyError:
            raise KeyError(f"div_type must be one of {[key for key in self.DIVERGENCE_MEASURES.keys()]} but is {div_type}.")

        return divergence_measure(pd_x, pd_y)
