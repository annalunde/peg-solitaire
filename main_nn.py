from critic_dict import CriticDict
from actor import Actor
from simworld import Environment
from critic_nn import CriticNN


def main():
    env = Environment(step_reward=0, final_reward=1, loser_penalty=0, boardsize=4, open_cells=[(2, 1)],
                      board_type="Diamond", track_history=False)
    actor = Actor(learning_rate=0.9, discount_factor=0.9,
                  eli_decay=0.9, epsilon=0.1)
    critic = CriticNN(dims=(15, 14, 3, 1), alpha=0.73,
                      eli_decay=0.9, gamma=0.8, shape="Diamond", size=4)

    for episode in range(1000):
        env.new_game()
        path = []
        cases = []
        targets = []
        print(f"Playing episode number {episode+1}")
        # critic.reset_eli_dict()
        actor.reset_eli_dict()
        while not env.game_is_finished():
            current_state = str(env.get_state())
            legal_actions = env.get_actions()
            action = actor.get_action(current_state, legal_actions)
            path.append((str(current_state), str(action)))
            reward = env.perform_action(action)

            td_err = critic.compute_td_err(
                current_state, env.get_state(), reward)
            critic.splitGD.update_td_error(td_err)

            cases.append(current_state)
            targets.append(critic.compute_target(reward, env.get_state()))
            critic.train(cases, targets)

            critic.update_eli_dict(str(current_state), 0)
            actor.update_eli_dict(str(current_state), str(action), 0)

            for i, sap in enumerate(reversed(path)):
                # critic.update_cases(sap[0],)
                #critic.splitgd.update_eli_dict(str(sap[0]), 1)
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
