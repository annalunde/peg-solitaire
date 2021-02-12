from critic_dict import CriticDict
from actor import Actor
from simworld import Environment
from critic_nn import CriticNN
from copy import deepcopy


def main():
    env = Environment(step_reward=0, final_reward=1, loser_penalty=0, boardsize=4, open_cells=[(2, 1)],
                      board_type="Diamond", track_history=False)
    actor = Actor(learning_rate=0.9, discount_factor=0.9,
                  epsilon_decay=0.995, epsilon=0.1, eli_decay=0.8)
    critic = CriticNN(dims=(16, 3, 1), alpha=0.73,
                      eli_decay=0.9, gamma=0.8)

    for episode in range(1000):
        env.new_game()
        path = []
        cases = []
        targets = []
        print(f"Playing episode number {episode+1}")
        critic.splitGD.reset_eli_dict()
        actor.reset_eli_dict()
        while not env.game_is_finished():
            state = deepcopy(env.get_state())
            legal_actions = env.get_actions()
            action = actor.get_action(state=state, legal_actions=legal_actions)
            path.append((str(state), str(action)))
            reward = env.perform_action(action=action)

            td_err = critic.compute_td_err(state=state, state_prime=env.get_state(), reward=reward)
            # critic.splitGD.update_td_error(td_err)
            #print("td_err", td_err)

            #critic.splitGD.update_eli_dict()
            actor.update_eli_dict(str(state), str(action), 0)
            #target = critic.compute_target(reward, env.get_state())
            #print("target", target)
            # cases.append(current_state)
            # for state in reversed(cases):
            critic.train(state, td_err)
            critic.splitGD.decay_eligibilites()

            for i, sap in enumerate(reversed(path)):
                actor.update_policy_dict(str(sap[0]), str(sap[1]), td_err)
                actor.update_eli_dict(str(sap[0]), str(sap[1]), 1)

    env.board.show_gameplay()

    env.new_game()

    print(f"Actor final epsilon: {actor.epsilon}")
    actor.epsilon = 0
    print("Attempting final gameplay to show you how smart I am now")
    while not env.game_is_finished():
        current_state = env.get_state()
        legal_actions = env.get_actions()
        action = actor.get_action(current_state, legal_actions)
        env.perform_action(action)
    env.board.visualize(0.3)


main()
