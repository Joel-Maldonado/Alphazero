from coach import Coach
# from tictactoe.tictactoe import TicTacToe as Game
# from tictactoe.tictactoe_network import NNetWrapper as nn
from connect4.connect4 import Connect4 as Game
from connect4.connect4_network import NNetWrapper as nn

args = {
    'num_iters': 1000,
    'num_eps': 10,              # Number of complete self-play games to simulate during a new iteration.
    'temp_threshold': 15,        #
    'update_threshold': 0.55,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'max_len_of_queue': 200000,    # Number of game examples to train the neural networks.
    'num_mcts_sims': 50,          # Number of games moves for MCTS to simulate.
    'arena_compare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,

    'checkpoint': './checkpoints/connect4/',
    'load_model': True,
    'load_folder_file': ('./checkpoints/connect4','best.pth.tar'),
    'num_iters_for_train_examples_history': 20
}


def main():
    print(f'Loading {Game.__name__}...')
    g = Game

    print(f'Loading {nn.__name__}...')
    nnet = nn(g)

    if args['load_model']:
        print('Loading checkpoint "%s/%s"...', args['load_folder_file'][0], args['load_folder_file'][1])
        nnet.load_checkpoint(args['load_folder_file'][0], args['load_folder_file'][1])
    else:
        print('Not loading a checkpoint! Starting from scratch...')

    print('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args['load_model']:
        print("Loading 'trainExamples' from file...")
        c.load_train_examples()

    print('Starting the learning process 🎉')
    c.learn()

if __name__ == "__main__":
    main()