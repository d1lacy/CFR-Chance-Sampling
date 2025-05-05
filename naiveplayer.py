from pypokerengine.players import BasePokerPlayer
import random as rand
import pprint
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate
from pypokerengine.utils.card_utils import gen_cards

# Naive Player that decides its action based on a Monte Carlo evaluation of the hand

class NaivePlayer(BasePokerPlayer):

  def declare_action(self, valid_actions, hole_card, round_state):
    
    win_prob = estimate_hole_card_win_rate(300, 2, gen_cards(hole_card), gen_cards(round_state["community_card"]))
    if win_prob >= 0.7 and len(valid_actions) ==3:
      action = valid_actions[2]['action']
    elif win_prob > 0.3:
      action = valid_actions[1]['action']
    else:
      action = valid_actions[0]['action']

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

def setup_ai():
  return NaivePlayer()