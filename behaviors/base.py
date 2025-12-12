from abc import ABC, abstractmethod

class BehaviorStrategy(ABC):
    @abstractmethod
    def decide_action(self, agent, view):
        pass
