from critic_dict import CriticDict
from actor import Actor
from simworld import Environment


def main():
    env = Environment(step_reward=10, final_reward=20, loser_penalty=-50, boardsize=4, open_cells=[(2, 1)],
                      board_type="Diamond", track_history=True)
    critic = CriticDict(learning_rate=0.1, eli_decay=0.7, discount_factor=0.7)
    actor = Actor(learning_rate=0.1, discount_factor=0.7, eli_decay=0.7, epsilon=0.5)

    for episode in range(1000):
        env.new_game()
        path = []
        print(f"Playing episode number {episode+1}")
        while not env.game_is_finished():
            current_state = env.get_state()
            legal_actions = env.get_actions()

            action = actor.get_action(current_state, legal_actions)
            reward = env.perform_action(action)
            path.append(current_state)

            td_err = critic.compute_td_err(current_state, action, reward)

            for i, state in enumerate(path):
                critic.update_eli_dict(state, td_err, i)
                critic.update_value_dict(state, td_err)
                actor.update_eli_dict(state, action, i)
                actor.update_policy_dict(state, action, td_err)



        critic.reset_eli_dict()
        actor.reset_eli_dict()

    env.board.show_gameplay()

    env.new_game()

    print(f"Actor final epsilon: {actor.epsilon}")
    actor.epsilon = 0
    print("Attempting final gameplay to show you how smart I am now")
    print(actor.policy_dict)
    while not env.game_is_finished():
        current_state = env.get_state()
        legal_actions = env.get_actions()
        action = actor.get_action(current_state, legal_actions)
        env.perform_action(action)
    env.board.show_gameplay()

main()
