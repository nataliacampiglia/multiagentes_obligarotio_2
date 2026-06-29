from abc import ABC, abstractmethod
from base.game import AgentID, ObsType, AlternatingGame
from numpy import ndarray
from gymnasium.spaces import Discrete, Text, Dict, Tuple
from pettingzoo.utils import agent_selector
from games.tictactoe import tictactoe_v3 as tictactoe
import numpy as np

import warnings
warnings.filterwarnings("ignore")


class TicTacToeAbs(AlternatingGame, ABC):

    def __init__(self, render_mode=''):
        super().__init__()
        self.env = tictactoe.raw_env(render_mode=render_mode)
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
        self.action_space = self.env.action_space
        self.agents = self.env.agents
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))

    def _update(self):
        self.rewards = self.env.rewards
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos
        self.agent_selection = self.env.agent_selection

    def reset(self):
        self.env.reset()
        self._update()

    def observe(self, agent: AgentID) -> ObsType:
        observation = self.env.observe(agent=agent)['observation']
        grid = np.sum(observation * [1, 2], axis=2)
        return grid

    def step(self, action):
        self.env.step(action)
        self._update()

    def available_actions(self):
        return self.env.board.legal_moves()

    def render(self):
        print("Player:", self.agent_selection)
        print("Board:")
        sq = np.array(self.env.board.squares).reshape((3, 3))
        for i in range(3):
            for j in range(3):
                if sq[i, j] == 0:
                    print(" . ", end="")
                elif sq[i, j] == 1:
                    print(" X ", end="")
                else:
                    print(" O ", end="")
            print()
        print()

    def clone(self):
        clone = self.__class__(render_mode=self.env.render_mode)
        clone.env.board.squares = self.env.board.squares.copy()
        clone.env.rewards = self.env.rewards.copy()
        clone.env.terminations = self.env.terminations.copy()
        clone.env.truncations = self.env.truncations.copy()
        clone.env.infos = clone.env.infos.copy
        clone.env.agent_selection = self.env.agent_selection
        clone._update()
        return clone

    @abstractmethod
    def eval(self, agent: AgentID) -> float:
        pass


class TicTacToe(TicTacToeAbs):
    """Eval basado en líneas abiertas (filas, columnas y diagonales sin fichas del rival)."""

    def eval(self, agent: AgentID) -> float:
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} is not part of the game.")
        if self.terminated():
            return self.rewards[agent]
        grid = self.observe(agent)
        E_agent    = self._eval(grid, 2)
        E_opponent = self._eval(grid, 1)
        return (E_agent - E_opponent) / 8.0

    def _eval(self, grid, player) -> float:
        rows = sum(int(all(grid[i] != player)) for i in range(3))
        cols = sum(int(all(grid.T[i] != player)) for i in range(3))
        diag1 = int(all(grid.diagonal() != player))
        diag2 = int(all(np.fliplr(grid).diagonal() != player))
        return rows + cols + diag1 + diag2

class TicTacToeEval(TicTacToeAbs):
    """
    Eval basado en amenazas: líneas con 2 fichas propias + 1 casilla vacía valen más.
    Por cada línea sin piezas del rival, suma owns² — una línea con 2 piezas propias vale 4,
    con 1 vale 1, vacía vale 0. Esto prioriza activamente completar líneas en lugar de solo mantenerlas abiertas.
    """

    def eval(self, agent: AgentID) -> float:
        if agent not in self.agents:
            raise ValueError(f"Agent {agent} is not part of the game.")
        if self.terminated():
            return self.rewards[agent]
        grid = self.observe(agent)
        E_agent    = self._eval(grid, 2)
        E_opponent = self._eval(grid, 1)    
        return (E_agent - E_opponent) / 8.0 

    def _eval(self, grid, player) -> float:
        lines = (
            [grid[i] for i in range(3)]          # filas
            + [grid.T[i] for i in range(3)]       # columnas
            + [grid.diagonal()]                    # diagonal principal
            + [np.fliplr(grid).diagonal()]         # diagonal secundaria
        )
        score = 0.0
        for line in lines:
            # cant de fichas de agente
            owns   = np.sum(line == player)
            # cant de vacios en la linea
            blanks = np.sum(line == 0)
            # cant de fichas del rival
            rival  = 3 - owns - blanks
            if rival == 0:
                # línea sin fichas del rival: pesa según cuántas propias hay
                score += owns ** 2
            # si hay fichas del rival en mi linea, no es buena la linea y no suma nada
        return score
