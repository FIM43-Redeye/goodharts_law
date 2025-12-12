from abc import ABC, abstractmethod
from typing import List

class Environment(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> List[str]:
        """Returns a list of capabilities this environment supports.
        Example: ['ground_truth', 'proxy_metric']
        """
        pass
