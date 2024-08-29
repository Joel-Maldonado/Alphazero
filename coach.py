import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import numpy as np
from tqdm.auto import tqdm

from arena import Arena
from mcts import MCTS



class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.mcts = MCTS(self.game, self.nnet, self.args)
        self.train_examples_history = []  # history of examples from args['num_iters_for_train_examples_history'] latest iterations
        self.skip_first_self_play = False  # can be overriden in loadTrainExamples()

    def execute_episode(self):
        """
        This function executes one episode of self-play, starting with player 1.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        board = self.game.get_initial_state()
        self.curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard = self.game.get_cannonical_state(board, self.curPlayer)
            temp = int(episodeStep < self.args['temp_threshold'])

            pi = self.mcts.get_action_prob(canonicalBoard, temp=temp)
            sym = self.game.get_symmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b, self.curPlayer, p, None])

            action = np.random.choice(len(pi), p=pi)
            board, self.curPlayer = self.game.get_next_state(board, action, self.curPlayer)

            r = self.game.get_game_outcome(board, self.curPlayer)

            if r != 0:
                return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args['num_iters'] + 1):
            # bookkeeping
            # log.info(f'Starting Iter #{i} ...')
            print(f'Starting Iter #{i} ...')
            # examples of the iteration
            if not self.skip_first_self_play or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args['max_len_of_queue'])

                for _ in tqdm(range(self.args['num_eps']), desc="Self Play"):
                    self.mcts = MCTS(self.game, self.nnet, self.args)  # reset search tree
                    iterationTrainExamples += self.execute_episode()

                # save the iteration examples to the history 
                self.train_examples_history.append(iterationTrainExamples)

            if len(self.train_examples_history) > self.args['num_iters_for_train_examples_history']:
                # log.warning(
                #     f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.train_examples_history)}")
                print(f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.train_examples_history)}")
                self.train_examples_history.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)  
            self.save_train_examples(i - 1)

            # shuffle examples before training
            trainExamples = []
            for e in self.train_examples_history:
                trainExamples.extend(e)
            shuffle(trainExamples)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')
            pmcts = MCTS(self.game, self.pnet, self.args)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.game, self.nnet, self.args)

            # log.info('PITTING AGAINST PREVIOUS VERSION')
            print('PITTING AGAINST PREVIOUS VERSION')
            arena = Arena(lambda x: np.argmax(pmcts.get_action_prob(x, temp=0)),
                          lambda x: np.argmax(nmcts.get_action_prob(x, temp=0)), self.game)
            pwins, nwins, draws = arena.play_games(self.args['arena_compare'])

            # log.info('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            print('NEW/PREV WINS : %d / %d ; DRAWS : %d' % (nwins, pwins, draws))
            if pwins + nwins == 0 or float(nwins) / (pwins + nwins) < self.args['update_threshold']:
                # log.info('REJECTING NEW MODEL')
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args['checkpoint'], filename='temp.pth.tar')
            else:
                # log.info('ACCEPTING NEW MODEL')
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename=self.get_checkpoint_file(i))
                self.nnet.save_checkpoint(folder=self.args['checkpoint'], filename='best.pth.tar')

    def get_checkpoint_file(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def save_train_examples(self, iteration):
        folder = self.args['checkpoint']
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.get_checkpoint_file(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.train_examples_history)
        f.closed

    def load_train_examples(self):
        modelFile = os.path.join(self.args['load_folder_file'][0], self.args['load_folder_file'][1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            # log.warning(f'File "{examplesFile}" with trainExamples not found!')
            print(f'File "{examplesFile}" with train_examples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            # log.info("File with trainExamples found. Loading it...")
            print("File with train_examples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.train_examples_history = Unpickler(f).load()
            # log.info('Loading done!')
            print('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skip_first_self_play = True
