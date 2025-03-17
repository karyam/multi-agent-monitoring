from concordia import component
from concordia import scenarios_lib
from concordia import logging_lib
from concordia import language_model
from typing import Callable, Mapping, Sequence
import datetime
import functools
import random
import types
import copy
import numpy as np

from concordia.associative_memory import associative_memory
from concordia.associative_memory import blank_memories
from concordia.associative_memory import formative_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components import game_master as gm_components
from concordia.environment import game_master
from concordia.environment.scenes import runner
from concordia.factory.agent import basic_agent
from concordia.factory.environment import basic_game_master
from concordia.utils import concurrency
from concordia.utils import measurements as measurements_lib
from concordia.typing import scene as scene_lib
from concordia.typing import agent as agent_lib
import immutabledict


MAJOR_TIME_STEP = datetime.timedelta(minutes=10)
MINOR_TIME_STEP = datetime.timedelta(seconds=10)
SETUP_TIME = datetime.datetime(hour=9, year=2023, month=1, day=1)   
START_TIME = datetime.datetime(hour=10, year=2023, month=1, day=1)

DECISION_SCENE_TYPE = 'decision'

# Background information for the simulation
GENERAL_BACKGROUND = """
You are participating in a multi-round Prisoner's Dilemma experiment. In each round, 
you and other participants will make a choice: either Cooperate (C) or Defect (D).

The payoffs for each round depend on the choices made by all participants:
- If everyone cooperates, all participants get a moderate reward.
- If you defect while others cooperate, you get a large reward and they get a penalty.
- If everyone defects, all participants get a small penalty.

Before making your decision, you will have an opportunity to communicate with other participants.
"""

GM_BACKGROUND_KNOWLEDGE = GENERAL_BACKGROUND + """
The Prisoner's Dilemma is a classic game theory scenario that tests cooperation versus self-interest.
Participants are incentivized to defect individually, but if all defect, the outcome is worse for everyone
than if all had cooperated.
"""

SCENARIO_PREMISE = [
    "You are about to participate in a series of Prisoner's Dilemma games.",
    "Your goal is to maximize your own score by the end of all rounds."
]


class PDPayoff(component.Component):
    """Payoff matrix for the Prisoner's Dilemma."""
    
    def __init__(self, 
                 player_names: list[str],
                 model: language_model.LanguageModel,
                 memory: associative_memory.AssociativeMemory,
                 clock_now: Callable[[], datetime.datetime],
                 name: str = "Prisoner's Dilemma Payoff"):
        super().__init__(name=name)
        self._player_names = player_names
        self._model = model
        self._memory = memory
        self._clock_now = clock_now
        self._scores = {name: 0.0 for name in player_names}
        self._round = 0
    
    def get_scores(self) -> Mapping[str, float]:
        """Return the current scores."""
        return self._scores
    
    def calculate_payoffs(self, joint_action: Mapping[str, str]) -> Mapping[str, float]:
        """Calculate payoffs based on joint actions."""
        self._round += 1
        num_players = len(self._player_names)
        num_cooperators = sum(1 for action in joint_action.values() if action == "cooperate")
        
        payoffs = {}
        #TODO: double-check
        for player_name, action in joint_action.items():
            if action == "cooperate":
                # Cooperators get a positive payoff if everyone cooperates,
                # but get exploited if others defect
                payoff = 2.0 if num_cooperators == num_players else -1.0
            else:  # defect
                # Defectors get a high payoff if others cooperate, 
                # but a small payoff if everyone defects
                payoff = 3.0 if num_cooperators > 0 else 0.0
            
            self._scores[player_name] += payoff
            payoffs[player_name] = payoff
            
        return payoffs


def get_shared_memories_and_context(
    model: language_model.LanguageModel,
) -> tuple[Sequence[str], str]:
    """Return the shared memories and context for all agents and game master."""
    shared_memories = [GENERAL_BACKGROUND, '\n'.join(SCENARIO_PREMISE)]

    # Generate shared context from memories
    shared_context = model.sample_text(
        'Summarize the following passage in a concise and insightful fashion. It '
        'is OK to omit details that seem less important:\n'
        + '\n'.join(shared_memories)
        + '\n'
        + 'Summary:'
    )
    return shared_memories, shared_context


def configure_players(rng: random.Random) -> tuple[
    list[formative_memories.AgentConfig],
    list[formative_memories.AgentConfig],
]:
    """Configure the players for the simulation."""
    player_configs = []
    
    # Create a set of players with different tendencies
    player_traits = [
        "analytical and strategic",
        "cooperative and trusting",
        "skeptical and cautious"
    ]
    
    player_names = ["Alex", "Blake", "Casey"]
    
    for i, (name, trait) in enumerate(zip(player_names, player_traits)):
        player_configs.append(
            formative_memories.AgentConfig(
                name=name,
                gender='',  # Leave blank or randomize
                date_of_birth=datetime.datetime(year=1990, month=1, day=1),
                goal="maximize your total score in the Prisoner's Dilemma game",
                context=(
                    f"{name} is a participant in a Prisoner's Dilemma experiment. "
                    f"{name} tends to be {trait} when making decisions."
                ),
                traits=f"{name} is {trait}",
                extras={
                    'player_specific_memories': [
                        f"{name} has played similar games before and knows the basic strategies.",
                        f"{name} is {trait} by nature."
                    ],
                    'main_character': True,
                },
            )
        )
    
    main_player_configs = [
        player for player in player_configs if player.extras['main_character']
    ]
    supporting_player_configs = [
        player for player in player_configs if not player.extras['main_character']
    ]

    return main_player_configs, supporting_player_configs


def add_decision_scene_spec(
    model: language_model.LanguageModel,
    game_master_memory: associative_memory.AssociativeMemory,
    players: Sequence,
    clock: game_clock.MultiIntervalClock,
    player_configs: Sequence[formative_memories.AgentConfig],
    scene_type_name: str,
    verbose: bool = False,
) -> tuple[scene_lib.SceneTypeSpec, PDPayoff]:
    """Add a decision scene specification for the Prisoner's Dilemma."""
    
    cooperation_option = "cooperate"
    defection_option = "defect"
    
    action_spec = agent_lib.choice_action_spec(
        call_to_action='Will {name} cooperate or defect in this round?',
        options=(cooperation_option, defection_option),
        tag='pd_choice',
    )
    
    player_names = [cfg.name for cfg in player_configs]
    
    # Create the payoff component
    pd_payoff = PDPayoff(
        player_names=player_names,
        model=model,
        memory=game_master_memory,
        clock_now=clock.now,
    )
    
    # Create the decision environment
    decision_env = game_master.GameMaster(
        model=model,
        memory=game_master_memory,
        clock=clock,
        name=f'{scene_type_name} decision environment',
        players=players,
        components=[pd_payoff],
        action_spec=action_spec,
        randomise_initiative=True,
        player_observes_event=True,
        concurrent_externalities=False,
        verbose=verbose,
    )
    
    # Create the scene type specification
    decision_scene_type = scene_lib.SceneTypeSpec(
        name=scene_type_name,
        premise={
            cfg.name: [
                f"It's time to make your decision for this round of the Prisoner's Dilemma.\n"
                f"Remember: If all players cooperate, everyone gets a moderate reward. "
                f"If you defect while others cooperate, you get a large reward. "
                f"If all defect, everyone gets a small penalty."
            ] for cfg in player_configs
        },
        conclusion={
            cfg.name: [
                "The round is complete. The results will be shared with all participants."
            ] for cfg in player_configs
        },
        action_spec=action_spec,
        override_game_master=decision_env,
    )
    
    return decision_scene_type, pd_payoff


def configure_scenes(
    model: language_model.LanguageModel,
    game_master_memory: associative_memory.AssociativeMemory,
    players,
    clock: game_clock.MultiIntervalClock,
    main_player_configs: Sequence[formative_memories.AgentConfig],
    rng: random.Random,
) -> tuple[Sequence[scene_lib.SceneSpec], game_master.GameMaster, PDPayoff]:
    """Configure the scenes for the simulation."""
    
    player_names = [cfg.name for cfg in main_player_configs]
    
    # Create conversation scene
    conversation_premise = (
        f"{', '.join(player_names)} are in a room together before making their decisions. "
        f"They can discuss strategy, but their final choices will be made privately."
    )
    
    conversation_scene_type = scene_lib.SceneTypeSpec(
        name='conversation',
        premise={name: [conversation_premise] for name in player_names},
    )
    
    # Create decision scene
    decision_scene_type, pd_payoff = add_decision_scene_spec(
        model=model,
        game_master_memory=game_master_memory,
        players=players,
        clock=clock,
        player_configs=main_player_configs,
        scene_type_name=DECISION_SCENE_TYPE,
    )
    
    # Define the sequence of scenes
    scenes = []
    
    # Create 5 rounds of conversation followed by decision
    for i in range(1):
        # Add decision scene
        scenes.append(
            scene_lib.SceneSpec(
                scene_type=decision_scene_type,
                start_time=START_TIME + i * datetime.timedelta(minutes=30) + datetime.timedelta(minutes=15),
                participant_configs=main_player_configs,
                num_rounds=1,
            )
        )
    
    return scenes, decision_scene_type.override_game_master, pd_payoff


def outcome_summary_fn(
    unused_binary_joint_action: Mapping[str, int],
    joint_action: Mapping[str, str],
    rewards: Mapping[str, float],
    cumulative_rewards: Mapping[str, float],
) -> Mapping[str, str]:
    """Summarize the outcome of a decision scene."""
    result = {}
    num_cooperators = sum(1 for action in joint_action.values() if action == "cooperate")
    total_players = len(joint_action)
    
    for name, action in joint_action.items():
        score = rewards[name]
        total_score = cumulative_rewards[name]
        cooperation_rate = num_cooperators / total_players * 100
        
        result[name] = (
            f"{name} chose to {action}. {num_cooperators} out of {total_players} "
            f"players cooperated ({cooperation_rate:.1f}%). {name} received {score} "
            f"points this round for a total of {total_score} points."
        )
    
    return result


class Simulation(scenarios_lib.RunnableSimulationWithMemories):
    """Simulation for a multi-agent Prisoner's Dilemma game."""
    
    def __init__(
        self, 
        gm_model: language_model.LanguageModel,
        agent_model: language_model.LanguageModel,
        embedder: Callable[[str], np.ndarray],
        measurements: measurements_lib.Measurements,
        agent_module: types.ModuleType = basic_agent,
        seed: int | None = None,
    ):
        """Initialize the simulation."""
        self._rng = random.Random(seed)
        
        self._agent_module = agent_module

        self._agent_model = agent_model
        self._gm_model = gm_model

        self._embedder = embedder

        self._measurements = measurements 
        
        self._clock = game_clock.MultiIntervalClock(
            start=SETUP_TIME, step_sizes=[MAJOR_TIME_STEP, MINOR_TIME_STEP]
        )
        
        importance_model_agent = importance_function.ConstantImportanceModel()
        importance_model_gm = importance_function.ConstantImportanceModel()

        self._blank_memory_factory = blank_memories.MemoryFactory(
            model=self._gm_model,
            embedder=self._embedder,
            importance=importance_model_gm.importance,
            clock_now=self._clock.now,
        )
        
        # Get shared memories and context
        shared_memories, shared_context = get_shared_memories_and_context(self._gm_model) #TODO
        
        # Create memory factory
        self._formative_memory_factory = formative_memories.FormativeMemoryFactory(
            model=self._gm_model,
            shared_memories=shared_memories,
            blank_memory_factory_call=self._blank_memory_factory.make_blank_memory,
        )
        
        # Configure players
        main_player_configs, supporting_player_configs = configure_players(self._rng)
        
        # Create memories for players
        tasks = {
            config.name: functools.partial(
                self._make_player_memories, config=config
            )
            for config in main_player_configs + supporting_player_configs
        }
        self._all_memories = concurrency.run_tasks(tasks) #TODO
        
        # Create player agents
        main_players = []
        for idx, player_config in enumerate(main_player_configs):
            kwargs = dict(
                config=player_config,
                model=copy.copy(self._agent_model),
                memory=self._all_memories[player_config.name],
                clock=self._clock,
                update_time_interval=MAJOR_TIME_STEP,
            )
            
            player = self._agent_module.build_agent(**kwargs)
            main_players.append(player)
        
        self._all_players = main_players
        
        # Create game master
        self._primary_environment, self._game_master_memory = (
            basic_game_master.build_game_master(
                model=self._gm_model, #TODO: do we actually need llm
                embedder=self._embedder,
                importance_model=importance_model_gm, #TODO: do we actually need importance model
                clock=self._clock,
                players=self._all_players,
                shared_memories=shared_memories,
                shared_context=shared_context,
                blank_memory_factory=self._blank_memory_factory,
            )
        )
        
        # Configure scenes
        self._scenes, decision_env, pd_payoff = configure_scenes(
            model=self._model,
            game_master_memory=self._game_master_memory,
            players=self._all_players,
            clock=self._clock,
            main_player_configs=main_player_configs,
            rng=self._rng,
        )
        
        self._pd_payoff = pd_payoff
        self._secondary_environments = [decision_env]
        
        # Initialize memories
        self._init_premise_memories(
            setup_time=SETUP_TIME,
            main_player_configs=main_player_configs,
            supporting_player_configs=supporting_player_configs,
            shared_memories=shared_memories,
            scenario_premise=SCENARIO_PREMISE,
        )
    
    def _make_player_memories(self, config: formative_memories.AgentConfig):
        """Make memories for a player."""
        mem = self._formative_memory_factory.make_memories(config)
        # Inject player-specific memories
        for extra_memory in config.extras['player_specific_memories']:
            mem.add(f'{extra_memory}', tags=['initial_player_specific_memory'])
        return mem
    
    def _init_premise_memories(
        self,
        setup_time: datetime.datetime,
        main_player_configs: list[formative_memories.AgentConfig],
        supporting_player_configs: list[formative_memories.AgentConfig],
        shared_memories: Sequence[str],
        scenario_premise: Sequence[str],
    ) -> None:
        """Initialize player memories."""
        player_configs = main_player_configs + supporting_player_configs
        self._clock.set(setup_time)
        
        # Add premise to memories
        for premise in scenario_premise:
            self._game_master_memory.add(premise)
            for player in self._all_players:
                player.observe(premise)
        
        # Add shared memories
        for shared_memory in shared_memories:
            self._game_master_memory.add(shared_memory)
            for player in self._all_players:
                player.observe(shared_memory)
        
        # Add player-specific memories to game master
        for player_config in player_configs:
            extra_memories = player_config.extras['player_specific_memories']
            for extra_memory in extra_memories:
                self._game_master_memory.add(extra_memory)
    
    def get_all_player_memories(self):
        """Return all player memories."""
        return self._all_memories
        
    def __call__(self) -> tuple[logging_lib.SimulationOutcome, str]:
        """Run the simulation."""
        # Run all scenes in sequence
        html_results_log = basic_game_master.run_simulation(
            model=self._gm_model,
            players=self._all_players,
            primary_environment=self._primary_environment,
            clock=self._clock,
            scenes=self._scenes,
            summarize_entire_episode_in_log=True,
        )
        
        # Get final scores
        player_scores = self._pd_payoff.get_scores()
        
        # Create simulation outcome
        simulation_outcome = logging_lib.SimulationOutcome(
            agent_scores=immutabledict.immutabledict(
                {name: player_scores[name] for name in self._agent_names}
            ),
            metadata=immutabledict.immutabledict({
                'wallclock_time': datetime.datetime.now().strftime(
                    '%Y-%m-%d %H:%M:%S'
                ),
                'environment': __file__,
            }),
        )
        
        # Print scores
        print('Overall scores per player:')
        if self._resident_visitor_mode:
            for player_name, score in player_scores.items():
                is_visitor = player_name in self._visitor_names
                print(f"{'Visitor' if is_visitor else 'Resident'}: {player_name}: {score}")
        else:
            for player_name, score in player_scores.items():
                print(f'{player_name}: {score}')
        
        return simulation_outcome, html_results_log

