from math import log, sqrt
import numpy as np

DEFAULT_EXPLORATION = sqrt(2)


def mean_reward(node, agent=None) -> float:
    agent = agent if agent is not None else node.agent
    if node.visits == 0:
        return 0

    agent_idx = node.game.agent_name_mapping[agent]
    return node.cum_rewards[agent_idx] / node.visits


def ucb(node, agent=None, C=DEFAULT_EXPLORATION) -> float:
    """
    La parte compartida entre MCTS e ISMCTS.

    Es la idea de ExploreAction:
        valor que vengo estimando + bonus por probar cosas poco visitadas

    O sea: si una accion parece buena, sube; si casi no la mire, tambien sube.
    """
    if node.visits == 0:
        return float('inf')

    parent_visits = max(node.parent.visits, 1)
    exploitation = mean_reward(node, agent)
    exploration = C * sqrt(log(parent_visits) / node.visits)
    return exploitation + exploration


def uct(node, agent=None):
    """
    UCT = elegir el hijo con mejor UCB.

    El parametro agent queda opcional porque el MCTS viejo ya tenia esa firma,
    pero si lo pasamos queda mas claro desde que punto de vista estamos eligiendo.
    """
    return max(node.children, key=lambda child: ucb(child, agent))


def unexpanded_actions(node, available_actions):
    expanded_actions = {child.action for child in node.children}
    return [action for action in available_actions if action not in expanded_actions]


def random_unexpanded_action(node, available_actions):
    actions = unexpanded_actions(node, available_actions)
    if not actions:
        return None
    return np.random.choice(actions)


def random_rollout_rewards(game, rollouts: int):
    rewards = np.zeros(len(game.agents))
    if rollouts <= 0:
        return rewards

    for _ in range(rollouts):
        rollout_game = game.clone()

        # Rollout random uniforme: simple, bruto, pero sirve como baseline.
        while not rollout_game.game_over():
            actions = rollout_game.available_actions()
            if len(actions) == 0:
                break

            action = np.random.choice(actions)
            rollout_game.step(action)

        rewards += np.array([rollout_game.reward(agent) for agent in rollout_game.agents])

    return rewards / rollouts


def backpropagate(node, rewards, agent):
    curr_node = node

    while curr_node is not None:
        curr_node.visits += 1
        curr_node.cum_rewards += rewards
        curr_node.value = mean_reward(curr_node, agent)
        curr_node = curr_node.parent


def best_child(node, agent):
    if not node.children:
        return None

    return max(
        node.children,
        key=lambda child: (mean_reward(child, agent), child.visits)
    )


def best_action_value(node, agent):
    child = best_child(node, agent)
    if child is None:
        return None, 0

    return child.action, mean_reward(child, agent)
