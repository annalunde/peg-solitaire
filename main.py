from critic_dict import CriticDict
from actor import Actor
from simworld import Environment
from math import exp


def play(env, actor):
    """
    Plays a single game within the environment, used for playing final game after learning phase
    """
    while not env.game_is_finished():
        current_state = env.get_state()
        legal_actions = env.get_actions()
        action = actor.get_action(current_state, legal_actions)
        print(f"Value of state action pair \n{current_state} \n{action}: {actor.get_policy(current_state, action)}")
        env.perform_action(action)
    return env


def main():
    wins = 0
    num_episodes = 1
    env = Environment(step_reward=1, final_reward=10, loser_penalty=-5, boardsize=4, open_cells=[(2, 1)],
                      board_type="Diamond", track_history=False)
    critic = CriticDict(learning_rate=0.5, eli_decay=0.9, discount_factor=0.9)
    actor = Actor(learning_rate=0.5, discount_factor=0.9, eli_decay=0.9, epsilon=1, epsilon_decay=exp(-1 / num_episodes))

    for episode in range(num_episodes):
        env.new_game()
        path = []
        print(f"Playing episode number {episode + 1}")
        while not env.game_is_finished():
            current_state = env.get_state()
            legal_actions = env.get_actions()
            action = actor.get_action(current_state, legal_actions)
            path.append((current_state, action))
            reward = env.perform_action(action)

            td_err = critic.compute_td_err(current_state, env.get_state(), reward)
            for i, state_action_pair in enumerate(path[::-1]):
                state, action = state_action_pair[0], state_action_pair[1]
                critic.update_eli_dict(state, i)
                critic.update_value_dict(state, td_err)
                print(f"Actor eli dict: {actor.eli_dict}")
                actor.update_eli_dict(state, action, i)
                print(f"Actor eli dict: {actor.eli_dict}")
                actor.update_policy_dict(state, action, td_err)
            print()

        #print("Value of first state-action pair:")
        #print(actor.get_policy([[1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1]], action=[(0, 3), (2, 1)]))

        # Quit after first success
        if env.board.is_won():
            wins += 1

        critic.reset_eli_dict()
        actor.reset_eli_dict()

    #env.board.show_gameplay()
    print(f"Total number of wins: {wins}")
    print(f"Actor final epsilon: {actor.epsilon}")
    print("Attempting final gameplay to show you how smart I am now")

    env.new_game()
    actor.epsilon = 0
    #env = play(env, actor)
    #env.board.show_gameplay()


main()
