from critic_dict import TableCritic
from critic_nn import CriticNN
from actor import Actor
from simworld import Environment
import yaml
import matplotlib.pyplot as plt
from copy import deepcopy

config = yaml.full_load(open("configs/5_triangle_table.yml"))
env_cfg = config["Environment"]
actor_cfg = config["Actor"]
critic_cfg = config["Critic_table"]
training_cfg = config["Training"]


def plot_learning(remaining_pegs):
    """
    Plots remaining pieces after each episode during a full run of training
    Should converge to one if the agent is learning
    """
    episode = [i for i in range(len(remaining_pegs))]
    plt.plot(episode, remaining_pegs)
    plt.xlabel("Episode number")
    plt.ylabel("Remaining pegs")
    plt.show()


def main(neural=True):
    """
    Sets the parameters for the Environment, Critic, and Actor according to the imported config file.
    Creates an environment where a predefined number of episodes can be performed.
    Instantiates an actor to keep track of the policy, and a critic to keep track of the value at each state
    Runs a predefined number of episodes creating a new board for each episode.
    For each episode, the actor and the critic are updated according to the Actor-Critic model.
    Finally, epsilon is set to zero, and the environment plays a game with the updated policy.
    """
    env = Environment(step_reward=env_cfg["step_reward"],
                      final_reward=env_cfg["final_reward"],
                      loser_penalty=env_cfg["loser_penalty"],
                      boardsize=env_cfg["boardsize"],
                      open_cells=env_cfg["open_cells"],
                      board_type=env_cfg["board_type"])
    critic_class = CriticNN if neural else TableCritic
    critic = critic_class(learning_rate=critic_cfg["learning_rate"],
                          eli_decay=critic_cfg["eli_decay"],
                          discount_factor=critic_cfg["discount_factor"],
                          dims=critic_cfg["dims"])
    actor = Actor(learning_rate=actor_cfg["learning_rate"],
                  discount_factor=actor_cfg["discount_factor"],
                  eli_decay=actor_cfg["eli_decay"],
                  epsilon=actor_cfg["epsilon"],
                  epsilon_decay=actor_cfg["epsilon_decay"])
    remaining_pegs = []

    for episode in range(training_cfg["number_of_episodes"]):
        env.new_game()
        path = []
        print(f"Playing episode number {episode + 1}")
        critic.reset_eli_dict()
        actor.reset_eli_dict()
        while not env.game_is_finished():
            current_state = deepcopy(env.get_state())
            legal_actions = env.get_actions()
            action = actor.get_action(
                state=current_state, legal_actions=legal_actions)
            path.append((str(current_state), str(action)))
            reward = env.perform_action(action=action)

            td_err = critic.compute_td_err(
                current_state=current_state, next_state=env.get_state(), reward=reward)

            if neural:
                # Previous states on the path are updated as well during the call to train() by eligibility traces
                critic.train(state=current_state, td_error=td_err)
                critic.update_eligs()

            # Update actor beliefs on SAPs for all pairs seen thus far in the episode
            for i, sap in enumerate(reversed(path)):
                actor.update_eli_dict(
                    state=str(sap[0]), action=str(sap[1]), i=i)
                actor.update_policy_dict(
                    state=str(sap[0]), action=str(sap[1]), td_err=td_err)
                if not neural:
                    critic.update_eligs(str(sap[0]), i)
                    critic.train(state=str(sap[0]), td_error=td_err)

        remaining_pegs.append(env.board.count_pieces())

    plot_learning(remaining_pegs)

    # Enable history tracking to visualize final game
    env.new_game(track_history=True)

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
