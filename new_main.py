from critic_dict import CriticDict
from actor import Actor
from simworld import Environment


def main():
    env = Environment(step_reward=10, final_reward=100, loser_penalty=-50, boardsize=4, open_cells=[(2, 1)],
                      board_type="Diamond", track_history=True)
    critic = CriticDict(learning_rate=0.1, eli_decay=0.5, discount_factor=0.7)
    actor = Actor(learning_rate=0.1, discount_factor=0.5,
                  eli_decay=0.7, epsilon=0.9)

    for episode in range(1000):
        env.new_game()
        path = []
        critic.reset_eli_dict()
        actor.reset_eli_dict()
        current_state = env.get_state()
        legal_actions = env.get_actions()
        action = actor.get_action(current_state, legal_actions)
        path.append((current_state, action))
        print(f"Playing episode number {episode+1}")
        while not env.game_is_finished():
            reward = env.perform_action(action)
            s_prime = env.get_state()
            legal_actions = env.get_actions()
            action_prime = actor.get_action(s_prime, legal_actions)

            td_err = critic.compute_td_err(current_state, s_prime, reward)
            path.append((s_prime, action_prime))
            for i, sap in enumerate(path):
                critic.update_eli_dict(sap[0], i)
                critic.update_value_dict(sap[0], td_err)
                actor.update_eli_dict(sap[0], sap[1], i)
                actor.update_policy_dict(sap[0], sap[1], td_err)

    env.board.show_gameplay()

    env.new_game()

    print(f"Actor final epsilon: {actor.epsilon}")
    actor.epsilon = 0
    print("Attempting final gameplay to show you how smart I am now")
    print("Actor policy learned so far:")
    print(actor.policy_dict)
    while not env.game_is_finished():
        current_state = env.get_state()
        legal_actions = env.get_actions()
        action = actor.get_action(current_state, legal_actions)
        env.perform_action(action)
    env.board.show_gameplay()


main()
