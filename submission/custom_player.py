from utils import *
import numpy as np
from pypokerengine.players import BasePokerPlayer


class CustomPlayer(BasePokerPlayer):
    def __init__(self):
        # load the strategy -- contains nodes (by key) with probabilities of each action
        self.strategy = load_strategy("strategy_5700.json")

  
    def declare_action(self, valid_actions,  hole_card, round_state):
        
        # get the information set key for the current state
        key = get_info_set_key(hole_card, round_state, len(valid_actions), self.uuid)

        # if the key is in the strategy, use the strategy to determine action
        if key in self.strategy:
            strategy = self.strategy[key]
            action_idx = np.random.choice([i for i in range(len(valid_actions))], p=strategy)
            action = valid_actions[action_idx]['action']

        # if the key is not in the strategy, use a simple strategy based on the estimated win
        else:
            est_win = int(list(key)[2])
            # if poor chance, fold
            if est_win < 3:
                action = 'fold'

            # if moderate chance and/or cant raise, call
            elif len(valid_actions) == 2 or est_win < 6:
                action = 'call'

            # if good chance and able, raise
            else:
                action = 'raise'

        return action
    

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

