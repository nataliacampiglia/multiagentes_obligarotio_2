from base.game import AgentID, ObsType
from numpy import ndarray
from gymnasium.spaces import Discrete, Text, Dict, Tuple
from pettingzoo.utils import agent_selector
from pettingzoo.classic import leduc_holdem_v4 as leduc
from base.game import AlternatingGame, AgentID, ActionType
import numpy as np
from functools import reduce

import warnings
warnings.filterwarnings("ignore")

class Leduc(AlternatingGame):

    def __init__(self, render_mode=''):
        super().__init__()
        self.env = leduc.raw_env(render_mode=render_mode)
        self.observation_spaces = self.env.observation_spaces
        self.action_spaces = self.env.action_spaces
        self.action_space = self.env.action_space
        self.agents = self.env.agents
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self.render_mode = render_mode
        self._hist = ''
        self._moves = ['c', 'r', 'f', 'k']

    def _update(self):
        self.rewards = self.env.rewards
        self.terminations = self.env.terminations
        self.truncations = self.env.truncations
        self.infos = self.env.infos
        self.agent_selection = self.env.agent_selection

    def observe(self, agent: AgentID) -> ObsType:
        state = self.env.env.game.get_state(self.env._name_to_int(agent))
        hand = state['hand'][1]
        public_card = '#' if state['public_card'] is None else state['public_card'][1]
        chips = '_'.join([str(x) for x in state['all_chips']])
        obs = hand + '_' + public_card + '_' + chips + '_' + self._hist
        return obs
    
    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        self.env.reset(seed, options)
        self._update()
        self._hist = str(self.env._name_to_int(self.agent_selection))
    
    def render(self) -> ndarray | str | list | None:
        return self.env.render()
    
    def step(self, action: ActionType) -> None:
        self._hist += self._moves[action] 
        self.env.step(action)
        self._update()

    def clone(self):
        import copy
        # deepcopy no funciona: raw_env usa EzPickle, cuyo __getstate__ solo guarda
        # los args del constructor y pierde todo el estado de reset(). Creamos una
        # instancia nueva y restauramos el estado manualmente.
        #
        # El clon tiene las cartas reales de ambos jugadores. 
        # Para CFR esto es correcto, la información oculta se maneja en observe(), no en el estado del juego.
        # Para ISMCTS usar sample_from_infoset(), que aleatoriza la mano del oponente.
        new_game = Leduc(render_mode=self.render_mode)
        if not hasattr(self.env, 'agent_selection'):
            # El juego nunca fue reseteado; no hay estado post-reset que copiar.
            new_game._hist = self._hist
            return new_game
        new_game.env.reset()  # inicializa terminations / rewards / etc. en raw_env
        # Estado interno del juego rlcard (cartas, apuestas, rondas)
        new_game.env.env.game = copy.deepcopy(self.env.env.game)
        new_game.env.env.timestep = self.env.env.timestep
        new_game.env.env.action_recorder = copy.deepcopy(self.env.env.action_recorder)
        # Estado del wrapper pettingzoo (agente activo, terminaciones, recompensas)
        new_game.env.agent_selection = self.env.agent_selection
        new_game.env.agents = list(self.env.agents)
        new_game.env.terminations = dict(self.env.terminations)
        new_game.env.truncations = dict(self.env.truncations)
        new_game.env.rewards = dict(self.env.rewards)
        new_game.env._cumulative_rewards = dict(self.env._cumulative_rewards)
        new_game.env.infos = copy.deepcopy(self.env.infos)
        new_game.env.next_legal_moves = list(self.env.next_legal_moves)
        new_game.env._last_obs = copy.deepcopy(self.env._last_obs)
        new_game._hist = self._hist
        new_game._update()
        return new_game

    def available_actions(self):
        if hasattr(self.env, 'next_legal_moves'):
            return list(self.env.next_legal_moves)
        # Fallback when pettingzoo reset() was never called (e.g. uninitialized clone)
        state = self.env.env.get_state(self.env.env.game.game_pointer)
        return list(state['legal_actions'])
    
    def sample_from_infoset(self, agent):
        import random as rnd
        clone = self.clone()
        game = clone.env.env.game
        my_idx = clone.env._name_to_int(agent)
        opp_idx = 1 - my_idx

        # cartas que el agente no puede ver: mano del oponente + mazo restante
        pool = [game.players[opp_idx].hand] + list(game.dealer.deck)
        rnd.shuffle(pool)

        game.players[opp_idx].hand = pool[0]
        game.dealer.deck = pool[1:]

        return clone

    def close(self):
        self.env.close()