from critic_dict import CriticDict
from actor import Actor
from simworld import Environment


def main():
    env = Environment(step_reward=0, final_reward=1, loser_penalty=0, boardsize=4, open_cells=[(2, 1)],
                      board_type="Diamond", track_history=True)
    critic = CriticDict(learning_rate=0.9, eli_decay=0.9, discount_factor=0.9)
    actor = Actor(learning_rate=0.9, discount_factor=0.9,
                  eli_decay=0.9, epsilon=0.1)

    for episode in range(1000):
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

            # UPDATE ELIG FIRST, THEN DECAYE
            for i, sap in enumerate(reversed(path)):
                critic.update_value_dict(str(sap[0]), td_err)
                critic.update_eli_dict(str(sap[0]), 1)
                actor.update_policy_dict(str(sap[0]), str(sap[1]), td_err)
                actor.update_eli_dict(str(sap[0]), str(sap[1]), 1)

    env.board.show_gameplay()

    env.new_game()

    print(f"Actor final epsilon: {actor.epsilon}")
    actor.epsilon = 0
    print("Attempting final gameplay to show you how smart I am now")
    # print(actor.policy_dict)
    while not env.game_is_finished():
        current_state = env.get_state()
        legal_actions = env.get_actions()
        action = actor.get_action(current_state, legal_actions)
        env.perform_action(action)
    env.board.show_gameplay()


main()
