from naiveplayer import NaivePlayer
from pypokerengine.api.game import setup_config, start_poker
from submission.custom_player import CustomPlayer


config = setup_config(max_round=500, initial_stack=1000, small_blind_amount=10)

config.register_player(name="NaivePlayer", algorithm=NaivePlayer())
config.register_player(name="CustomPlayer", algorithm=CustomPlayer())

game_result = start_poker(config, verbose=1)

