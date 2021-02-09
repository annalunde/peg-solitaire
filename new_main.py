from critic_dict import CriticDict
from actor import Actor
from simworld import Environment
import yaml
import matplotlib.pyplot as plt
import time

config = yaml.full_load(open("configs/5_triangle_table.yml"))
env_cfg = config["Environment"]
actor_cfg = config["Actor"]
critic_cfg = config["Critic_table"]
training_cfg = config["Training"]

def plot_learning(remaining_pegs):
    episode = [i for i in range(len(remaining_pegs))]
    plt.plot(episode, remaining_pegs)
    plt.show()

def main():
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
                      board_type= env_cfg["board_type"],
                      track_history=env_cfg["track_history"])
    critic = CriticDict(learning_rate=critic_cfg["learning_rate"],
                        eli_decay=critic_cfg["eli_decay"],
                        discount_factor=critic_cfg["discount_factor"])
    actor = Actor(learning_rate=actor_cfg["learning_rate"],
                  discount_factor=actor_cfg["discount_factor"],
                  eli_decay=actor_cfg["eli_decay"],
                  epsilon=actor_cfg["epsilon"],
                  epsilon_decay=actor_cfg["epsilon_decay"])
    remaining_pegs = []

<<<<<<< HEAD
    for episode in range(1000):
=======
    for episode in range(training_cfg["number_of_episodes"]):
>>>>>>> 9a9584d5b882c37a8a1f971e9d27d7084b05efaf
        env.new_game()
        path = []
        print(f"Playing episode number {episode+1}")
        critic.reset_eli_dict()
        actor.reset_eli_dict()
        while not env.game_is_finished():
            current_state = str(env.get_state())
            legal_actions = env.get_actions()
            action = actor.get_action(current_state, legal_actions)
            path.append((str(current_state), str(action)))
            reward = env.perform_action(action)

            td_err = critic.compute_td_err(
                current_state, env.get_state(), reward)

            critic.update_eli_dict(str(current_state), 0)
            actor.update_eli_dict(str(current_state), str(action), 0)

            # UPDATE ELIG FIRST, THEN DECAY
            for i, sap in enumerate(reversed(path)):
                critic.update_value_dict(str(sap[0]), td_err)
                critic.update_eli_dict(str(sap[0]), 1)
                actor.update_policy_dict(str(sap[0]), str(sap[1]), td_err)
                actor.update_eli_dict(str(sap[0]), str(sap[1]), 1)

        remaining_pegs.append(env.board.count_pieces())

    #env.board.show_gameplay()
    plot_learning(remaining_pegs)

    env.new_game()

    #print(f"Actor final epsilon: {actor.epsilon}")
    actor.epsilon = 0
    print("Attempting final gameplay to show you how smart I am now")
    # print(actor.policy_dict)
    while not env.game_is_finished():
        current_state = env.get_state()
        legal_actions = env.get_actions()
        action = actor.get_action(current_state, legal_actions)
        env.perform_action(action)
    env.board.visualize(0.1)


main()
