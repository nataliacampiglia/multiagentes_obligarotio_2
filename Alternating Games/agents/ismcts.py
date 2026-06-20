from base.game import AlternatingGame, AgentID, ActionType
from base.agent import Agent
from agents.tree_search import (
    backpropagate,
    best_action_value,
    random_rollout_rewards,
    random_unexpanded_action,
    ucb,
    unexpanded_actions,
)
import numpy as np
from typing import Callable


class ISMCTSNode:
    """
    Nodo del arbol de ISMCTS.

    Parecido al nodo de MCTS, pero guardamos tambien info_state.
    La diferencia conceptual es esa: el arbol no representa solo estados reales,
    sino lo que el agente puede distinguir (information sets).
    """
    def __init__(
        self,
        parent: 'ISMCTSNode',
        game: AlternatingGame,
        action: ActionType,
        info_state,
    ):
        self.parent = parent
        self.game = game
        self.action = action
        self.info_state = info_state
        self.children = []
        self.explored_children = 0
        self.visits = 0
        self.value = 0
        self.cum_rewards = np.zeros(len(game.agents))
        self.agent = self.game.agent_selection


class InformationSetMCTS(Agent):
    def __init__(
        self,
        game: AlternatingGame,
        agent: AgentID,
        simulations: int = 100,
        rollouts: int = 10,
        sample_from_infoset: Callable[[AlternatingGame, AgentID], AlternatingGame] = None,
        information_state_key: Callable[[AlternatingGame, AgentID], object] = None,
    ) -> None:
        """
        La idea es que no se busca desde "el estado real" si no lo conozco.
        Busco desde mi information set I, o sea desde los estados que podrian ser
        reales segun lo que vi. En cada simulacion sampleo un estado posible
        compatible con lo que sabe el agente y corro MCTS sobre eso.

        Si el juego todavia no tiene informacion oculta, sample_from_infoset
        termina siendo clone y esto se comporta parecido a MCTS normal.
        """
        super().__init__(game=game, agent=agent)
        self.simulations = simulations
        self.rollouts = rollouts
        self.sample_from_infoset = sample_from_infoset or self.default_sample_from_infoset
        self.information_state_key = information_state_key or self.default_information_state_key

    def action(self) -> ActionType:
        a, _ = self.ismcts()
        return a

    def ismcts(self) -> (ActionType, float):
        # Estado raiz de la busqueda.
        # En ISMCTS, antes de buscar sampleamos un estado concreto compatible
        # con lo que el agente sabe. Si no hay informacion oculta, es clone().
        root_game = self.sample_from_infoset(self.game, self.agent)
        root = ISMCTSNode(
            parent=None,
            game=root_game,
            action=None,
            info_state=self.information_state_key(root_game, self.agent),
        )

        for _ in range(self.simulations):
            # -----------------------
            # Sample from infoset
            # -----------------------
            # Nuevo estado posible: "supongamos que el mundo oculto era este".
            # Ejemplo mental: en poker, reparto las cartas ocultas de una forma
            # posible segun mi mano y lo publico.
            game = self.sample_from_infoset(self.game, self.agent)
            node = root

            # -----------------------
            # Selection
            # -----------------------
            # Bajo por el arbol mientras pueda elegir acciones ya expandidas
            # que sean legales en este estado posible.
            node = self.select_node(node, game)

            # -----------------------
            # Expansion
            # -----------------------
            # Si llegue a un infoset donde todavia falta probar alguna accion,
            # creo un hijo nuevo para esa accion.
            node = self.expand_node(node, game)

            # -----------------------
            # Simulation / Rollout
            # -----------------------
            # Desde el estado posible actual, juego random hasta terminar.
            rewards = self.rollout(game)

            # -----------------------
            # Backpropagation
            # -----------------------
            # La recompensa del rollout se propaga por los infosets visitados.
            self.backprop(node, rewards)

        # Luego de todas las simulaciones, elegimos la accion mas prometedora
        # desde el infoset raiz.
        return self.action_selection(root)

    def default_sample_from_infoset(self, game: AlternatingGame, agent: AgentID) -> AlternatingGame:
        """
        Hook para juegos con informacion imperfecta.

        Si el juego implementa alguno de estos metodos, genial: lo usamos.
        Si no, clonamos y listo. Para TicTacToe/NoccaNocca no hay cartas ocultas
        ni estado privado, entonces esto alcanza.
        """
        # Si el ambiente sabe samplear un estado compatible con el infoset,
        # dejamos que lo haga. Este seria el caso posta para ISMCTS.
        if hasattr(game, 'sample_from_infoset'):
            return game.sample_from_infoset(agent)

        # Alias clasicos/legacy por si algun juego lo trae con nombre teorico.
        if hasattr(game, 'sample_determinization'):
            return game.sample_determinization(agent)

        if hasattr(game, 'determinize'):
            return game.determinize(agent)

        # Fallback para juegos de informacion perfecta.
        return game.clone()

    def default_information_state_key(self, game: AlternatingGame, agent: AgentID):
        """
        Identifica el infoset.

        Primero le preguntamos al juego, porque en juegos tipo poker el infoset
        depende de cartas propias + historial publico. Si el juego no sabe,
        usamos la observacion como clave aproximada.
        """
        # Mejor caso: el juego define exactamente cual es el information set.
        if hasattr(game, 'information_state_key'):
            return game.information_state_key(agent)

        # Tambien sirve si devuelve una estructura, la convertimos a hashable.
        if hasattr(game, 'information_state'):
            return self.hashable(game.information_state(agent))

        acting_agent = game.agent_selection

        # PettingZoo-style: si hay diccionario de observaciones, usamos eso.
        if hasattr(game, 'observations') and acting_agent in game.observations:
            return (acting_agent, self.hashable(game.observations[acting_agent]))

        # NoccaNocca en clones no siempre trae observations listas,
        # pero si tiene board podemos usarlo como representacion.
        if hasattr(game, 'board') and hasattr(game.board, 'squares'):
            return (acting_agent, self.hashable(game.board.squares))

        # Ultimo intento: llamar observe. Si tampoco funciona, usamos repr
        # para al menos tener una clave estable-ish y no romper la busqueda.
        try:
            return (acting_agent, self.hashable(game.observe(acting_agent)))
        except Exception:
            return (acting_agent, repr(game))

    def hashable(self, value):
        # Algunas observaciones vienen como numpy/list/dict.
        # Para usarlas como key del infoset las pasamos a tuplas.
        if isinstance(value, np.ndarray):
            return tuple(value.flatten().tolist())
        if isinstance(value, dict):
            return tuple(sorted((key, self.hashable(val)) for key, val in value.items()))
        if isinstance(value, list):
            return tuple(self.hashable(val) for val in value)
        if isinstance(value, tuple):
            return tuple(self.hashable(val) for val in value)
        if isinstance(value, set):
            return tuple(sorted(self.hashable(val) for val in value))
        return value

    def select_node(self, node: ISMCTSNode, game: AlternatingGame) -> ISMCTSNode:
        """
        Selection: bajamos por el arbol usando solo hijos compatibles.

        En ISMCTS no alcanza con decir "este hijo existe": tambien tiene que ser
        legal en el estado posible que sampleamos ahora.
        """
        curr_node = node

        while not game.game_over():
            # Guardamos una copia del estado posible en el nodo.
            # No es el estado real, es el estado sampleado para esta simulacion.
            curr_node.game = game.clone()
            curr_node.agent = game.agent_selection

            # A(I): acciones posibles desde el infoset/estado sampleado.
            actions = list(game.available_actions())
            if len(actions) == 0:
                break

            # Si hay alguna accion legal que todavia no tiene hijo,
            # selection termina aca y expansion se encarga.
            if unexpanded_actions(curr_node, actions):
                break

            # En ISMCTS puede pasar que un hijo exista en el infoset,
            # pero no sea legal en este estado posible puntual.
            compatible_children = [
                child for child in curr_node.children
                if child.action in actions
            ]
            if not compatible_children:
                break

            # ExploreAction(I): usamos UCB sobre los hijos compatibles.
            child = self.select_child(compatible_children, game.agent_selection)

            # Avanzamos tambien el estado sampleado, para que node y game
            # sigan hablando del mismo camino.
            game.step(child.action)
            curr_node = child

        return curr_node

    def expand_node(self, node: ISMCTSNode, game: AlternatingGame) -> ISMCTSNode:
        """
        Expansion: si todavia queda alguna accion de A(I) sin hijo, agregamos uno.

        Si no se puede expandir (terminal, sin acciones, o todo ya expandido),
        devolvemos el mismo nodo y seguimos con rollout desde ahi.
        """
        if game.game_over():
            return node

        # El nodo queda asociado al estado donde vamos a expandir.
        node.game = game.clone()
        node.agent = game.agent_selection

        # Acciones legales de este estado posible.
        available_actions = list(game.available_actions())
        if len(available_actions) == 0:
            return node

        # Acciones validas que aun no fueron expandidas.
        action = random_unexpanded_action(node, available_actions)
        if action is None:
            return node

        # Elegimos una accion nueva para probar.
        # Aplicamos la transicion en el estado posible actual.
        game.step(action)

        # Creamos el hijo que representa haber elegido esa accion.
        child = ISMCTSNode(
            parent=node,
            game=game.clone(),
            action=action,
            info_state=self.information_state_key(game, self.agent),
        )
        node.children.append(child)
        return child

    def select_child(self, children, agent: AgentID) -> ISMCTSNode:
        """
        Primero probamos acciones no visitadas; despues UCB.
        Medio lo mismo que en MCTS, solo que aca miramos hijos compatibles.
        """
        # En el teorico de MCTS esto es "probar cada accion una vez"
        # antes de confiar en UCB.
        unvisited = [child for child in children if child.visits == 0]
        if unvisited:
            return np.random.choice(unvisited)

        # UCB balancea explotar lo que viene dando bien y explorar lo poco visto.
        return max(children, key=lambda child: ucb(child, agent))

    def rollout(self, game: AlternatingGame):
        # No tocamos el estado de selection/expansion; rollout va en copia.
        # Guardamos recompensa para todos los agentes, igual que en MCTS.
        # Promedio sobre los rollouts.
        return random_rollout_rewards(game, self.rollouts)

    def backprop(self, node: ISMCTSNode, rewards):
        # Visito cada infoset del camino una vez mas.
        # Acumulo el vector de recompensas.
        # Valor desde el punto de vista del agente que esta pensando.
        # Subo al padre hasta llegar a la raiz.
        backpropagate(node, rewards, self.agent)

    def action_selection(self, node: ISMCTSNode) -> (ActionType, float):
        # BestAction(I): elegimos la accion con mejor recompensa media.
        # En empate, preferimos la mas visitada.
        return best_action_value(node, self.agent)


ISMCTS = InformationSetMCTS
