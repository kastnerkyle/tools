import numpy as np
import copy
import cPickle

class MemoizeMutable(object):
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}

    def __call__(self, *args, **kwds):
        str = cPickle.dumps(args, 1) + cPickle.dumps(kwds, 1)
        if not self.memo.has_key(str):
            self.memo[str] = self.fn(*args, **kwds)
        else:
            pass
        return self.memo[str]


def softmax(x):
    assert len(x.shape) == 1
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class TreeNode(object):
    def __init__(self, prior_prob, parent):
        self.parent = parent
        self.Q_ = 0.
        self.P_ = float(prior_prob)
        # action -> tree node
        self.children_ = {}
        self.n_visits_ = 0

    def expand(self, actions_and_probs):
        for action, prob in actions_and_probs:
            if action not in self.children_:
                self.children_[action] = TreeNode(prob, self)

    def is_leaf(self):
        return self.children_ == {}

    def is_root(self):
        return self.parent is None

    def _update(self, value):
        self.n_visits_ += 1
        # not tracking W directly
        # original update is
        # n_visits += 1
        # W += v
        # Q = W / n_visits
        # so,
        # the old W = Q * (n_visits - 1)
        # new W = old W + v
        # new Q = new W / n_visits
        # plugging in new W
        # new Q = (old W + v) / n_visits
        # plugging in old W
        # new Q = (Q * (n_visits - 1) + v)/n_visits
        # new_Q = (Q * n_visits - Q + v)/n_visits
        # new_Q = Q * n_visits/n_visits - Q/n_visits + v/n_visits
        # new_Q = Q - Q/n_visits + v/n_visits
        # new_Q = Q + (v - Q) / n_visits
        # new_Q += (v - Q) / n_visits
        self.Q_ += (value - self.Q_) / float(self.n_visits_)

    def update(self, value):
        if self.parent != None:
            # negative in the original code due to being the opposing player
            self.parent.update(value)
        self._update(value)

    def get_value(self, c_puct):
        self.U_ = c_puct * self.P_ * np.sqrt(float(self.parent.n_visits_)) / float(1. + self.n_visits_)
        return self.Q_ + self.U_

    def get_best(self, c_puct):
        best = max(self.children_.iteritems(), key=lambda x: x[1].get_value(c_puct))
        return best


class MCTS(object):
    def __init__(self, state_manager, c_puct=1.4, n_playout=1000, random_state=None):
        if random_state is None:
            raise ValueError("Must pass random_state object")
        self.random_state = random_state
        self.root = TreeNode(1., None)
        # state manager must, itself have *NO* state / updating behavior
        # internally. Otherwise we need deepcopy() in get_move_probs
        self.state_manager = state_manager
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.tree_subs_ = []
        self.warn_at_ = 10000

    def playout(self, state):
        node = self.root
        while True:
            if node.is_leaf():
                break
            action, node = node.get_best(self.c_puct)
            state = self.state_manager.get_next_state(state, action)
        winner, score, end = self.state_manager.is_finished(state)
        if not end:
            # uniform prior probs
            actions = self.state_manager.get_valid_actions(state)
            action_space = self.state_manager.get_action_space()
            probs = np.ones((len(actions))) / float(len(actions))
            actions_and_probs = list(zip(actions, probs))
            node.expand(actions_and_probs)
        value = self.state_manager.rollout_from_state(state)
        # negative here
        node.update(value)
        return None

    def get_action_probs(self, state, temp=1E-3):
        # low temp -> nearly argmax
        for n in range(self.n_playout):
            self.playout(state)

        act_visits = [(act, node.n_visits_) for act, node in self.root.children_.items()]
        if len(act_visits) == 0:
            return None, None
        actions, visits = zip(*act_visits)
        action_probs = softmax(1. / temp * np.log(visits))
        return actions, action_probs

    def sample_action(self, state, temp=1E-3, add_noise=True,
                      dirichlet_coeff1=0.25, dirichlet_coeff2=0.3):
        vsz = len(self.state_manager.get_action_space())
        act_probs = np.zeros((vsz,))
        acts, probs = self.get_action_probs(state, temp)
        if acts == None:
            return acts, probs
        act_probs[list(acts)] = probs
        if add_noise:
            act = self.random_state.choice(acts, p=(1. - dirichlet_coeff1) * probs + dirichlet_coeff1 * self.random_state.dirichlet(dirichlet_coeff2 * np.ones(len(probs))))
        else:
            act = self.random_state.choice(acts, p=probs)
        return act, act_probs

    def get_action(self, state):
        vsz = len(self.state_manager.get_action_space())
        act_probs = np.zeros((vsz,))
        # temp doesn't matter for argmax
        acts, probs = self.get_action_probs(state, temp=1.)
        if acts == None:
            return acts, probs
        act_probs[list(acts)] = probs
        maxes = np.max(act_probs)
        opts = np.where(act_probs == maxes)[0]
        if len(opts) > 1:
            # choose the one with the highest win score if equal?
            # if 2 options are *exactly* equal, just choose 1 at random
            self.random_state.shuffle(opts)
        act = opts[0]
        return act, act_probs

    def update_tree_root(self, action):
        if action in self.root.children_:
            self.tree_subs_.append((self.root, self.root.children_[action]))
            if len(self.tree_subs_) > self.warn_at_:
                print("WARNING: Over {} tree_subs_ detected, watch memory".format(self.warn_at_))
                # only print the warning a few times
                self.warn_at_ = 10 * self.warn_at_
            self.root = self.root.children_[action]
            self.root.parent = None
        else:
            raise ValueError("Action argument {} neither in root.children_ {} and not == -1 (reset)".format(self.root.children_.keys()))

    def reconstruct_tree(self):
        # walk the list back to front, putting parents back in place
        # should reconstruct tree while still preserving counts...
        # this might be a bad idea for large state spaces
        for pair in self.tree_subs_[::-1]:
            self.root.parent = pair[0]
            self.root = pair[0]
        self.tree_subs_ = []

    def reset_tree(self):
        print("Resetting tree")
        self.root = TreeNode(1., None)
        self.tree_subs_ = []
