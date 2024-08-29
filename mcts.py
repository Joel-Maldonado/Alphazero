import numpy as np
import math

EPS = 1e-8

class MCTS():
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times state s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.s_outcomes = {}  # stores game.get_game_outcome for state s
        self.s_valid_actions = {}  # stores game.get_valid_actions for state s

    def get_action_prob(self, canonicalBoard, temp=1):
        for _ in range(self.args['num_mcts_sims']):
            self.search(canonicalBoard)

        s = self.game.state_to_string(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.get_action_size())]

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs
    
    def search(self, cannonical_state):
        s = self.game.state_to_string(cannonical_state)

        if s not in self.s_outcomes:
            self.s_outcomes[s] = self.game.get_game_outcome(cannonical_state, 1)

        # terminal node    
        if self.s_outcomes[s] != 0:
            return -self.s_outcomes[s]

        if s not in self.Ps:
            # leaf node
            self.Ps[s], v = self.nnet.predict(cannonical_state)
            valid_actions = self.game.get_valid_actions(cannonical_state)
            self.Ps[s] = self.Ps[s] * valid_actions
            sum_of_Ps = np.sum(self.Ps[s])
            if sum_of_Ps > 0:
                self.Ps[s] /= sum_of_Ps
            else:
                print("All valid moves were masked, settings all valid moves to be equally probably. Check if your NNet architecture is insufficient or you've get overfitting!")
                self.Ps[s] = self.Ps[s] + valid_actions
                self.Ps[s] /= np.sum(self.Ps[s])

            self.s_valid_actions[s] = valid_actions
            self.Ns[s] = 0

            return -v

        valid_actions = self.s_valid_actions[s]
        cur_best = -float('inf')
        best_act = -1

        for a in range(self.game.get_action_size()):
            if valid_actions[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args['cpuct'] * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args['cpuct'] * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_state, next_player = self.game.get_next_state(cannonical_state, a, 1)
        next_state = self.game.get_cannonical_state(next_state, next_player)

        v = self.search(next_state)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
