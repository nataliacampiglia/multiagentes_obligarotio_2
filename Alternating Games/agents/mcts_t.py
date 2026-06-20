from base.game import AlternatingGame, AgentID, ActionType
from base.agent import Agent
import numpy as np
from typing import Callable
from agents.tree_search import (
    backpropagate,
    best_action_value,
    random_rollout_rewards,
    random_unexpanded_action,
    uct,
)

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

        # Acciones validas desde este estado
        available_actions = list(node.game.available_actions())

        # Acciones validas que aun no fueron expandidas
        action = random_unexpanded_action(node, available_actions)
        if action is None:
            return

        # Elegimos una acción nueva para expandir
        child_game = node.game.clone()


        # aplicamos la transición del juego.
        # ŝτ+1 ∼ T(. | ŝτ, âτ) 
        child_game.step(action)

        # Creamos el nuevo nodo y lo agregamos al arbol
        # InitializeNode(ŝτ)
        child = MCTSNode(parent=node, game=child_game, action=action)
        node.children.append(child)

    def rollout(self, node):
        # TODO
        # implement rollout policy
        # for i in range(self.rollouts): 
        #     play random game and record average rewards
        # Copia del estado del nodo para no modificar el arbol.
        # Promedio de recompensas de todos los rollouts.
        return random_rollout_rewards(node.game, self.rollouts)

    def backprop(self, node, rewards):
        # TODO
        # cumulate rewards and visits from node to root navigating backwards through parent
        # while τ > t do
        #     Update(Q, ŝτ, âτ)
        backpropagate(node, rewards, self.agent)

    def action_selection(self, node: MCTSNode) -> (ActionType, float):
        # TODO
        # hint: return action of child with max value 
        # other alternatives could be considered

        # πt ← BestAction(st)
        action, value = best_action_value(node, self.agent)

        # at ∼ πt
        return action, value
