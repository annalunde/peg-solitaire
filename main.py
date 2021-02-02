from critic_dict import Critic_Dict
from actor import Actor
from simworld import Environment

def main():
  env = Environment(step_reward=1, final_reward=2, loser_penalty=-5, boardsize=4, open_cells=[(2,1)], board_type="Diamond", track_history=False)
  critic = Critic_Dict(learning_rate=0.2, eli_decay=0.2, discount_factor=0.2)
  actor = Actor(learning_rate=0.2, discount_factor=0.2, eli_decay=0.2, epsilon=0.1)

  for e in range(10):
    env.new_game()
    path = []
    while not env.game_is_finished():
      current_state = env.get_state()
      legal_actions = env.board.get_legal_actions()
      print(current_state)
      print(legal_actions)

      action = actor.get_action(current_state, legal_actions)
      print(action)
      reward = env.perform_action(action)
      print(reward)
      path.append(env.get_state())
      print(current_state.get_value())

      td_err = critic.compute_td_err(current_state, action, reward)
      for i, state in enumerate(path):
        critic.update_value_dict(state,td_err)
        critic.update_eli_dict(state, td_err, i)
        actor.update_policy_dict(state, action, td_err)
        actor.update_eli_dict(state, action,i)

    critic.reset_eli_dict()
    actor.reset_eli_dict()

main()
