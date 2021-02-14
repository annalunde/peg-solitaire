

class Critic:
    """
    Abstract class represent a generic critic.
    Subclasses such as neural nets and table critics implements methods
    """

    def __init__(self, learning_rate, eli_decay, discount_factor):
        self.learning_rate = learning_rate
        self.eli_decay = eli_decay
        self.discount_factor = discount_factor

    def compute_td_err(self, current_state, next_state, reward):
        """
        TD error describes the difference between reward + next state and the current state.
        Formula: r + discount_factor*V(s') - V(s)
        :param current_state: list(list(int))
        :param next_state: list(list(int))
        :param reward: int
        """
        pass

    def train(self, state, td_error):
        """
        Trains the critic on a new observation based on td_error
        """
        pass

    def reset_eli_dict(self):
        """
        Resets eligibilities (done before a new episode)
        """
        pass

    def update_eligs(self, *args):
        """
        Updates eligibility for a single step
        """
        pass
