STATE_MAX = 50

class RightPolicyStateManager(object):
    """
    A state manager with a goal state and action space of 2
    """
    def __init__(self, goal_state, random_state, rollout_limit=1000):
        self.rollout_limit = rollout_limit
        self.goal_state = goal_state
        self.random_state = random_state

    def get_next_state(self, state, action):
        if action == 1:
            return state + 1
        elif action == 0:
            return state - 1

    def get_action_space(self):
        # go down, 0
        # go up, 1
        return list(range(2))

    def get_valid_actions(self, state):
        if state > 0 and state < STATE_MAX:
            return list(range(2))
        elif state == 0:
            return [1]
        elif state == STATE_MAX:
            return [0]

    def get_init_state(self):
        # start in the worst state
        return 0
        #return self.random_state.randint(0, STATE_MAX)

    def rollout_fn(self, state):
        # can define custom rollout function
        return self.random_state.choice(self.get_valid_actions(state))

    def score(self, state):
        # if these numbers are big, it tends to run slower

        # example of custom finish, score
        # sparse / goal discovery reward
        #return 1. if state == self.goal_state else 0.
        # distance / goal conditioned reward
        return 1. if state == self.goal_state else -(1. / self.goal_state) * (self.goal_state - state)

    def is_finished(self, state):
        # if this check is slow
        # can rewrite as _is_finished
        # then add
        # self.is_finished = MemoizeMutable(self._is_finished)
        # to __init__ instead

        # return winner, score, end
        # winner normally in [-1, 0, 1]
        # if it's one player, can just use [0, 1] and it's fine
        # score arbitrary float value
        # end in [True, False]
        return (1, 1., True) if state == self.goal_state else (0, 0., False)

    def rollout_from_state(self, state):
        # example rollout function
        s = state
        w, sc, e = self.is_finished(state)
        if e:
            return self.score(s)

        c = 0
        while True:
            a = self.rollout_fn(s)
            s = self.get_next_state(s, a)

            e = self.is_finished(s)
            c += 1
            if e:
                return self.score(s)
            if c > self.rollout_limit:
                # can also return different score if rollout limit hit
                return self.score(s)


if __name__ == "__main__":
    from puct_mcts import MCTS, MemoizeMutable
    import numpy as np
    mcts_random = np.random.RandomState(1110)
    state_random = np.random.RandomState(11)
    exact = True

    state_man = RightPolicyStateManager(STATE_MAX, state_random)
    mcts = MCTS(state_man, n_playout=1000, random_state=mcts_random)
    state = mcts.state_manager.get_init_state()
    winner, score, end = mcts.state_manager.is_finished(state)
    states = [state]
    while True:
        if not end:
            if not exact:
                a, ap = mcts.sample_action(state, temp=temp, add_noise=noise)
            else:
                a, ap = mcts.get_action(state)

            for i in mcts.root.children_.keys():
                print(i, mcts.root.children_[i].__dict__)
                print("")
            mcts.update_tree_root(a)
            state = mcts.state_manager.get_next_state(state, a)
            states.append(state)
            print(states)
            winner, score, end = mcts.state_manager.is_finished(state)
        if end:
            print(states[-1])
            print("Ended")
            mcts.reconstruct_tree()
            break
