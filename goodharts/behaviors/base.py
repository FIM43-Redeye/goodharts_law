from abc import ABC, abstractmethod

class BehaviorStrategy(ABC):
    @property
    @abstractmethod
    def requirements(self) -> list[str]:
        """Returns a list of requirements this behavior needs from the environment.
        Example: ['ground_truth'] or ['proxy_metric']
        """
        pass

    @abstractmethod
    def decide_action(self, agent, view):
        pass
