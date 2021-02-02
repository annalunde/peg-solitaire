from critic_dict import CriticDict
from actor import Actor
from simworld import Environment


def main():
    #her er test
    env = Environment(step_reward=0, final_reward=1, loser_penalty=0, boardsize=4, open_cells=[(2, 1)],
                      board_type="Diamond", track_history=True)
    critic = CriticDict(learning_rate=1, eli_decay=1, discount_factor=1)
    actor = Actor(learning_rate=1, discount_factor=1, eli_decay=1, epsilon=0.1)

    for episode in range(2000):
        env.new_game()
        path = []
        print(f"Playing episode number {episode+1}")
        while not env.game_is_finished():
            current_state = env.get_state()
            legal_actions = env.get_actions()
            path.append(current_state)
            action = actor.get_action(current_state, legal_actions)
            reward = env.perform_action(action)

            td_err = critic.compute_td_err(
                current_state, env.get_state(), reward)

            for i, state in enumerate(path):
                critic.update_eli_dict(state, i)
                critic.update_value_dict(state, td_err)
                actor.update_eli_dict(state, action, i)
                actor.update_policy_dict(state, action, td_err)

            if env.board.is_won():
                break
        if env.board.is_won():
            break
        critic.reset_eli_dict()
        actor.reset_eli_dict()

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
