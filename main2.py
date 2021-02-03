from critic_dict import CriticDict
from actor import Actor
from simworld import Environment


def main():
    env = Environment(step_reward=0, final_reward=10, loser_penalty=-10, boardsize=4, open_cells=[(2, 1)],
                      board_type="Diamond", track_history=True)
    critic = CriticDict(
        learning_rate=0.9, eli_decay=0.8, discount_factor=0.9)
    actor = Actor(learning_rate=0.9, discount_factor=0.9,
                  eli_decay=0.9, epsilon=0.1)

    for episode in range(1000):
        env.new_game()
        critic.reset_eli_dict()
        actor.reset_eli_dict()
        path = []
        print(f"Playing episode number {episode+1}")
        current_state = str(env.get_state())
        legal_actions = env.get_actions()
        action = actor.get_action(current_state, legal_actions)
        path.append((str(current_state), str(action)))
        k = 0

        while not env.game_is_finished():
            print("print current: ", current_state)

            reward = env.perform_action(action)
            s_prime = env.get_state()

            # Sjekk current og s'
            print("print current k: ", path[k][0])
            print("print s'prime: ", s_prime)

            actor.update_eli_dict(path[k][0], action, 0)

            td_err = critic.compute_td_err(
                path[k][0], s_prime, reward)
            critic.update_eli_dict(path[k][0], 0)

            for sap in reversed(path):
                critic.update_value_dict(sap[0], td_err)
                critic.update_eli_dict(sap[0], 1)
                actor.update_policy_dict(sap[0], sap[1], td_err)
                actor.update_eli_dict(sap[0], sap[1], 1)

            current_state = env.get_state()
            legal_actions = env.get_actions()
            action = actor.get_action(current_state, legal_actions)
            path.append((str(current_state), str(action)))
            k += 1

        # print("td_err", td_err)
        # print("elig", actor.eli_dict[(str(env.get_state()), str(action))])

    # env.board.show_gameplay()

    env.new_game()

    print(f"Actor final epsilon: {actor.epsilon}")
    actor.epsilon = 0.2
    print("Attempting final gameplay to show you how smart I am now")
    # print(actor.policy_dict)
    while not env.game_is_finished():
        current_state = env.get_state()
        # print("current", current_state)
        legal_actions = env.get_actions()
        # print("legal acts", legal_actions)
        # for a in legal_actions:
        # print("a's value", a)
        # print(actor.policy_dict[(str(current_state), str(a))])
        # print('c´s value', critic.get_value(current_state))
        action = actor.get_action(current_state, legal_actions)
        # print("chosen action", action)
        env.perform_action(action)
    env.board.show_gameplay()


main()
