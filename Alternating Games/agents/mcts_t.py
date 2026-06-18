from base.game import AlternatingGame, AgentID, ActionType
from base.agent import Agent
from math import log, sqrt
import numpy as np
from typing import Callable

class MCTSNode:
    def __init__(self, parent: 'MCTSNode', game: AlternatingGame, action: ActionType):
        self.parent = parent
        self.game = game
        self.action = action
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.value = 0
        self.cum_rewards = np.zeros(len(game.agents))
        self.agent = self.game.agent_selection

def ucb(node, C=sqrt(2)) -> float:
    """
    Corresponde a ExploreAction(ŝτ) del algoritmo.

    Fórmula del algoritmo:
        Q(s,a) + C * sqrt( ln N(s) / N(s,a) )

    En este código:
        node                     = hijo que representa aplicar una acción a
        node.parent              = estado padre s
        node.visits              = N(s,a)
        node.parent.visits       = N(s)
        node.cum_rewards / visits = Q(s,a)
    """
    agent_idx = node.game.agent_name_mapping[node.agent]

    exploitation = node.cum_rewards[agent_idx] / node.visits
    exploration = C * sqrt(log(node.parent.visits)/node.visits)

    return exploitation + exploration

def uct(node: MCTSNode, agent: AgentID) -> MCTSNode:
    """
    UCT elige el hijo con mayor UCB.
    Esto implementa:
        âτ ← ExploreAction(ŝτ)
    cuando el nodo ya tiene todos sus hijos expandidos.
    """
    child = max(node.children, key=ucb)
    return child

class MonteCarloTreeSearch(Agent):
    def __init__(self, game: AlternatingGame, agent: AgentID, simulations: int=100, rollouts: int=10, selection: Callable[[MCTSNode, AgentID], MCTSNode]=uct) -> None:
        """
        Parameters:
            game: alternating game associated with the agent
            agent: agent id of the agent in the game
            simulations: number of MCTS simulations (default: 100)
            rollouts: number of MC rollouts (default: 10)
            selection: tree search policy (default: uct)
        """
        super().__init__(game=game, agent=agent)
        self.simulations = simulations
        self.rollouts = rollouts
        self.selection = selection
        
    def action(self) -> ActionType:
        a, _ = self.mcts()
        return a

    def mcts(self) -> (ActionType, float):

        # Observe current state s^t
        root = MCTSNode(parent=None, game=self.game, action=None)

        # for k simulations do
        for i in range(self.simulations):

            # τ ← t
            node = root

            # ŝτ ← st
            node.game = self.game.clone()

            #print(i)
            #node.game.render()

            # -----------------------
            # Selection
            # -----------------------
            # while ŝτ is non-terminal and ŝτ-node exists in tree do
            #     âτ ← ExploreAction(ŝτ)
            #     ŝτ+1 ∼ T(. | ŝτ, âτ)
            #     r̂τ ← R(...)
            #     τ ← τ + 1
            # select_node baja por el árbol
            print('--selection--')
            node = self.select_node(node=node)

            # ----------------------------
            # EXPANSION
            # ----------------------------
            # if ŝτ-node does not exist in tree then
            #     InitializeNode(ŝτ)
            #
            # Se agrega un hijo nuevo al árbol (expande).
            print('--expansion--')
            self.expand_node(node)

            # -----------------------
            # Simulation / Rollout
            # -----------------------
            print('--rollout--')
            rewards = self.rollout(node)

            # ----------------------------
            # BACKPROPAGATION
            # ----------------------------
            # while τ > t do
            #     τ ← τ - 1
            #     Update(Q, ŝτ, âτ)
            # Actualizamos visitas y recompensas desde el nodo hasta la raíz.
            print('--backprop--')
            self.backprop(node, rewards)

        #print('root childs')
        #for child in root.children:
        #    print(child.action, child.cum_rewards / child.visits)


        # Luego de k simulaciones, elegimos la acción real.
        # πt ← BestAction(st)
        # at ∼ πt
        action, value = self.action_selection(root)

        return action, value

    def backprop(self, node, rewards):
        # TODO
        # cumulate rewards and visits from node to root navigating backwards through parent
        curr_node = node

        # while τ > t do
        while curr_node is not None:
            # Update(Q, ŝτ, âτ)
            curr_node.visits += 1
            curr_node.cum_rewards += rewards

            agent_idx = curr_node.game.agent_name_mapping[self.agent]
            curr_node.value = curr_node.cum_rewards[agent_idx] / curr_node.visits

            # τ ← τ − 1
            curr_node = curr_node.parent

    def rollout(self, node):
        rewards = np.zeros(len(self.game.agents))
        # TODO
        # implement rollout policy
        # for i in range(self.rollouts): 
        #     play random game and record average rewards
        if self.rollouts <= 0:
            return rewards

        for _ in range(self.rollouts):
            # Copia del estado del nodo para no modificar el arbol
            game = node.game.clone()

            # Jugamos aleatoriamente hasta que termine el juego
            while not game.game_over():
                actions = game.available_actions()
                if len(actions) == 0:
                    break

                action = np.random.choice(actions)
                game.step(action)

            rewards += np.array([game.reward(agent) for agent in game.agents])

        # Promedio de recompensas de todos los rollouts.
        rewards = rewards / self.rollouts
        return rewards

    def select_node(self, node: MCTSNode) -> MCTSNode:
        """
        Baja por el arbol mientras existan hijos.
        Si todavía hay hijos no explorados, elige el siguiente hijo no explorado.
        Si todos fueron explorados, usa UCT/UCB.
        """
        curr_node = node
        # while ŝτ is non-terminal and ŝτ-node exists in tree do
            #     âτ ← ExploreAction(ŝτ)
            #     ŝτ+1 ∼ T(. | ŝτ, âτ)
            #     r̂τ ← R(...)
            #     τ ← τ + 1
        while curr_node.children:
            # Aun hay hijos que no fueron seleccionados, hacemos esto antes de comenzar a aplicar UCB.
            if curr_node.explored_children < len(curr_node.children):
                curr_node = curr_node.children[curr_node.explored_children]
                curr_node.parent.explored_children += 1

            # todos los hijos ya fueron explorados, ahora usamos UCT:
            # Q(s,a) + C sqrt(ln N(s) / N(s,a))
            else:
                # set curr_node to a child using the selection function
                # âτ ← ExploreAction(ŝτ)
                agent = curr_node.game.agent_selection
                curr_node = self.selection(curr_node, agent)
        return curr_node

    def expand_node(self, node) -> None:
        # TODO
        # if the game is not terminated: 
        #    play an available action in node
        #    create a new child node and add it to node children
        if node.game.game_over():
            return

        # Acciones que ya tienen un hijo en el arbol.
        expanded_actions = {child.action for child in node.children}

        # Acciones validas desde este estado
        available_actions = list(node.game.available_actions())

        # Acciones validas que aun no fueron expandidas
        unexplored_actions = [action for action in available_actions if action not in expanded_actions]

        if not unexplored_actions:
            return

        # Elegimos una acción nueva para expandir
        action = np.random.choice(unexplored_actions)
        child_game = node.game.clone()


        # aplicamos la transición del juego.
        # ŝτ+1 ∼ T(. | ŝτ, âτ) 
        child_game.step(action)

        # Creamos el nuevo nodo y lo agregamos al arbol
        # InitializeNode(ŝτ)
        child = MCTSNode(parent=node, game=child_game, action=action)
        node.children.append(child)

    def action_selection(self, node: MCTSNode) -> (ActionType, float):
        action: ActionType = None
        value: float = 0
        # TODO
        # hint: return action of child with max value 
        # other alternatives could be considered

        if not node.children:
            return action, value

        agent_idx = node.game.agent_name_mapping[self.agent]

        # πt ← BestAction(st)
        child = max(
            node.children,
            key=lambda child: (
                child.cum_rewards[agent_idx] / child.visits if child.visits > 0 else 0,
                child.visits
            )
        )

        # at ∼ πt
        action = child.action
        if child.visits > 0:
            value = child.cum_rewards[agent_idx] / child.visits

        return action, value
