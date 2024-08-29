import numpy as np
from mcts import MCTS
# from tictactoe.tictactoe import TicTacToe as Game
# from tictactoe.tictactoe_network import NNetWrapper as nn
from connect4.connect4 import Connect4 as Game
from connect4.connect4_network import NNetWrapper as nn
from main import args

args = {
    'num_mcts_sims': 25,          # Number of games moves for MCTS to simulate.
    'cpuct': 1,
}

def main():
    game = Game
    nnet = nn(game)
    nnet.load_checkpoint('./checkpoints/connect4', 'best.pth.tar')
    mcts = MCTS(game, nnet, args)

    # Play against bot
    state = game.get_initial_state()
    curPlayer = 1

    probs = mcts.get_action_prob(state, temp=0)

    while game.get_game_outcome(state, curPlayer) == 0:
        Game.visualize_state(state)

        if curPlayer == 1:
            print('Player 1\'s Turn')
            valids = game.get_valid_actions(game.get_cannonical_state(state, curPlayer))
            print(f'Valid actions: {np.where(valids[:-1] == 1)[0]}')
            while True:
                action = input('Enter your move: ')
                if action.isdigit():
                    if valids[int(action)] != 0:
                        action = int(action)
                        break
                    continue
        else:
            print('Bot\'s Turn')
            probs = mcts.get_action_prob(state, temp=0)
            valids = game.get_valid_actions(game.get_cannonical_state(state, curPlayer))
            action = np.argmax(probs * valids)
            if action == game.get_action_size():
                print('No valid actions left!')
                break
            print(f'Bot move: {action}')

        state, curPlayer = game.get_next_state(state, action, curPlayer)

    Game.visualize_state(state)


    outcome = game.get_game_outcome(state, curPlayer)
    if outcome == 1:
        print('Player wins!')
    elif outcome == -1:
        print('Bot wins!')
    else:
        print('Draw!')



if __name__ == "__main__":
    main()