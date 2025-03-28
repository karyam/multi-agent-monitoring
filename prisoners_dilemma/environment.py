# configure the scenes here
# configure payoff ?
# configure game master

class Environment:
    def __init__(self):
        self.game_master = GameMaster()
        self.scenes = []
        self.players = []
    
    def run(self):
        # run scenes here
        # equivalent to run_scenes in runner.py ?
        for _ in range()