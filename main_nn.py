from critic_dict import TableCritic
from actor import Actor
from simworld import Environment
from critic_nn import CriticNN
from copy import deepcopy
import matplotlib.pyplot as plt


def plot_learning(remaining_pegs):
    """
    Plots remaining pieces after each episode during a full run of training
    Should converge to one if the agent is learning
    """
    episode = [i for i in range(len(remaining_pegs))]
    plt.plot(episode, remaining_pegs)
    plt.show()


def main():
    env = Environment(step_reward=1, final_reward=50, loser_penalty=-100, boardsize=4, open_cells=[(2, 1)],
                      board_type="Diamond")
    actor = Actor(learning_rate=0.9, discount_factor=0.9,
                  epsilon_decay=0.995, epsilon=0.1, eli_decay=0.8)
    critic = CriticNN(dims=(16, 3, 1), learning_rate = 0.01, eli_decay=0.995, discount_factor= 0.8)

    remaining_pegs = []  # For plotting learning convergence
    neural = True

    for episode in range(100):
        env.new_game()
        path = []
        print(f"Playing episode number {episode + 1}")
        critic.reset_eli_dict()
        actor.reset_eli_dict()
        while not env.game_is_finished():
            current_state = deepcopy(env.get_state())
            legal_actions = env.get_actions()
            action = actor.get_action(state=current_state, legal_actions=legal_actions)
            path.append((str(current_state), str(action)))
            reward = env.perform_action(action=action)

            td_err = critic.compute_td_err(current_state=current_state, next_state=env.get_state(), reward=reward)

            if neural:
                # Previous states on the path are updated as well during the call to train() by eligibility traces
                critic.train(state=current_state, td_error=td_err)
                critic.update_eligs()

            # Update actor beliefs on SAPs for all pairs seen thus far in the episode
            for i, sap in enumerate(reversed(path)):
                actor.update_eli_dict(state=str(sap[0]), action=str(sap[1]), i=i)
                actor.update_policy_dict(state=str(sap[0]), action=str(sap[1]), td_err=td_err)
                if not neural:
                    critic.update_eligs(str(sap[0]), i)
                    critic.train(state=str(sap[0]), td_error=td_err)

        remaining_pegs.append(env.board.count_pieces())

    plot_learning(remaining_pegs)

    env.new_game(track_history=True)  # Enable history tracking to visualize final game

    print(f"Actor final epsilon: {actor.epsilon}")
    actor.epsilon = 0  # Set exploration to 0
    print("Attempting final gameplay to show you how smart I am now")
    while not env.game_is_finished():
        current_state = env.get_state()
        legal_actions = env.get_actions()
        action = actor.get_action(current_state, legal_actions)
        env.perform_action(action)
    env.board.visualize(0.3)


main()
