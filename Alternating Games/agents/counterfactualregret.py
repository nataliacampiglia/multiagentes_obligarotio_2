import numpy as np
from numpy import ndarray
from base.game import AlternatingGame, AgentID, ObsType
from base.agent import Agent

class Node():

    def __init__(self, game: AlternatingGame, obs: ObsType) -> None:
        self.game = game
        self.agent = game.agent_selection
        self.obs = obs
        self.num_actions = self.game.num_actions(self.agent)
        self.cum_regrets = np.zeros(self.num_actions)
        self.curr_policy = np.full(self.num_actions, 1/self.num_actions)
        self.sum_policy = self.curr_policy.copy()
        self.learned_policy = self.curr_policy.copy()
        self.niter = 1

    def regret_matching(self):
        # TODO
        positive_regrets = np.maximum(self.cum_regrets, 0)

        if positive_regrets.sum() > 0:
            self.curr_policy = positive_regrets / positive_regrets.sum()
        else:
            self.curr_policy = np.full(self.num_actions, 1 / self.num_actions)

        self.sum_policy += self.curr_policy
        self.niter += 1

        self.learned_policy = self.sum_policy / self.sum_policy.sum()
        
    
    def update(self, utility, node_utility, probability) -> None:
        # update 
        for a in range(self.num_actions):
            regret = utility[a] - node_utility
            self.cum_regrets[a] += probability * regret
        # regret matching policy
        self.regret_matching()  

    def policy(self):
        return self.learned_policy

class CounterFactualRegret(Agent):

    def __init__(self, game: AlternatingGame, agent: AgentID) -> None:
        super().__init__(game, agent)
        self.node_dict: dict[ObsType, Node] = {}

    def action(self):
        try:
            node = self.node_dict[self.game.observe(self.agent)]
            a = np.argmax(np.random.multinomial(1, node.policy(), size=1))
            return a
        except:
            #raise ValueError('Train agent before calling action()')
            print('Node does not exist. Playing random.')
            return np.random.choice(self.game.available_actions())
    
    def train(self, niter=1000):
        for _ in range(niter):
            _ = self.cfr()

    def cfr(self):
        game = self.game.clone()
        utility: dict[AgentID, float] = dict()
        for agent in self.game.agents:
            game.reset()
            probability = np.ones(game.num_agents)
            utility[agent] = self.cfr_rec(game=game, agent=agent, probability=probability)

        return utility 

    def cfr_rec(self, game: AlternatingGame, agent: AgentID, probability: ndarray):
        # TODO
        """
        CFR recursivo.

        agent:
            agente para el cual estamos calculando regrets en esta pasada.

        probability:
            reach probability de cada agente hasta este nodo.
            Ejemplo para 2 agentes:
                probability[0] = probabilidad de que agent_0 llegue hasta aquí
                probability[1] = probabilidad de que agent_1 llegue hasta aquí
        """

        # Paso base, si el juego terminó, devuelvo utilidad del agente entrenado
        if game.terminated():
            return game.reward(agent)

        current_agent = game.agent_selection
        obs = game.observe(current_agent)

        # Obtener o crear nodo
        if obs not in self.node_dict:
            self.node_dict[obs] = Node(game, obs)

        node = self.node_dict[obs]

        # Política actual del nodo
        policy = node.curr_policy

        num_actions = node.num_actions
        action_utilities = np.zeros(num_actions)

        node_utility = 0 #remove

        # índice del agente actual
        current_agent_idx = game.agents.index(current_agent)

        # Explorar cada acción posible
        for action in range(num_actions):
            next_game = game.clone()
            next_game.step(action)

            next_probability = probability.copy()
            next_probability[current_agent_idx] *= policy[action]

            action_utilities[action] = self.cfr_rec(
                game=next_game,
                agent=agent,
                probability=next_probability
            )

            node_utility += policy[action] * action_utilities[action]

        # Actualizar regrets solo si el nodo pertenece al agente que estoy entrenando
        if current_agent == agent:
            # Probabilidad de llegada contrafactual:
            # producto de las probabilidades de los otros agentes
            counterfactual_reach_prob = 1.0

            for i, other_agent in enumerate(game.agents):
                if other_agent != agent:
                    counterfactual_reach_prob *= probability[i]

            node.update(
                utility=action_utilities,
                node_utility=node_utility,
                probability=counterfactual_reach_prob
            )

        return node_utility
        
