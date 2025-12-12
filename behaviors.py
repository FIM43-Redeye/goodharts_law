import numpy as np
from abc import ABC, abstractmethod
from constants import CellType


class BehaviorStrategy(ABC):
    @abstractmethod
    def decide_action(self, agent, view):
        pass


class GreedyFoodSeeker(BehaviorStrategy):
    def decide_action(self, agent, view):
        center = agent.sight_radius
        best_moves = []
        safe_moves = []

        for dy in range(-agent.sight_radius, agent.sight_radius + 1):
            for dx in range(-agent.sight_radius, agent.sight_radius + 1):
                if np.sqrt(dx ** 2 + dy ** 2) > agent.sight_radius:
                    continue
                if dx == 0 and dy == 0:
                    continue

                cell = view[center + dy, center + dx]
                step_x = np.sign(dx)
                step_y = np.sign(dy)
                immediate_cell = view[center + step_y, center + step_x]

                if immediate_cell == CellType.WALL:
                    continue

                if cell == CellType.FOOD:
                    best_moves.append((step_x, step_y))
                elif cell == CellType.EMPTY and immediate_cell == CellType.EMPTY:
                    safe_moves.append((step_x, step_y))

        if best_moves:
            min_dist = float('inf')
            best_move = (0, 0)
            for dx, dy in best_moves:
                for food_dy in range(-agent.sight_radius, agent.sight_radius + 1):
                    for food_dx in range(-agent.sight_radius, agent.sight_radius + 1):
                        if view[center + food_dy, center + food_dx] == CellType.FOOD:
                            dist = np.sqrt((food_dx - dx) ** 2 + (food_dy - dy) ** 2)
                            if dist < min_dist:
                                min_dist = dist
                                best_move = (dx, dy)
            return best_move
        elif safe_moves:
            index = np.random.randint(0, len(safe_moves))
            return safe_moves[index]

        return 0, 0
