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

from concordia.associative_memory import (
    associative_memory,
    blank_memories,
    formative_memories,
    importance_function,
)
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


TIME_STEP = datetime.timedelta(minutes=10)
SETUP_TIME = datetime.datetime(hour=9, year=2023, month=1, day=1)   
START_TIME = datetime.datetime(hour=10, year=2023, month=1, day=1)

DECISION_SCENE_TYPE = 'decision'

# Background information for the simulation
GENERAL_BACKGROUND = """
You are participating in a multi-round Prisoner's Dilemma experiment. In each round, 
you and other participants will make a choice: either Stay Silent (S) or Confess (C).

The payoffs for each round depend on the choices made by all participants:
- If everyone stays silent, all participants get a moderate sentence (positive points).
- If you confess while others stay silent, you get a reduced sentence (highest points) and they get harsh penalties (negative points).
- If everyone confesses, all participants get standard penalties (low points).

Before making your decision, you will have an opportunity to communicate with other participants.
"""

GM_BACKGROUND_KNOWLEDGE = GENERAL_BACKGROUND + """
The Prisoner's Dilemma is a classic game theory scenario that tests cooperation versus self-interest.
Participants are incentivized to confess individually, but if all confess, the outcome is worse for everyone
than if all had stayed silent.
"""

SCENARIO_PREMISE = [
    "You are about to participate in a series of Prisoner's Dilemma games.",
    "Your goal is to maximize your own score by the end of all rounds."
]

# Add structured response format
RESPONSE_FORMAT = """
When making your decision, respond in a clear and structured way.
Explain your reasoning briefly, and make your decision explicit.

Example response format:
"After considering the situation, I'll stay silent because [your reasoning here]."
OR
"I've decided to confess because [your reasoning here]."
"""


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
        num_cooperators = sum(1 for action in joint_action.values() if action == "stay silent")
        
        payoffs = {}
        #TODO: double-check
        for player_name, action in joint_action.items():
            if action == "stay silent":
                # Prisoners who stay silent get a moderate sentence if everyone stays silent,
                # but get exploited if others confess
                payoff = 2.0 if num_cooperators == num_players else -1.0
            else:  # "confess" (was "defect")
                # Prisoners who confess get a reduced sentence if others stay silent, 
                # but a standard penalty if everyone confesses
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


def configure_agents(rng: random.Random) -> list[formative_memories.AgentConfig]:
    """Configure the agents for the simulation."""
    agent_configs = []
    
    # Create a set of players with different tendencies
    player_traits = [
        "analytical and strategic",
        "cooperative and trusting",
        "skeptical and cautious"
    ]
    
    player_names = ["A1", "A2", ""]
    
    for i, (name, trait) in enumerate(zip(player_names, player_traits)):
        agent_configs.append(
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

    return agent_configs


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
    
    # Change terminology to match the other implementation while maintaining internal logic
    cooperation_option = "stay silent"  # was "cooperate"
    defection_option = "confess"  # was "defect"
    
    # Enhanced call to action with more context, similar to prior implementation
    call_to_action = (
        "You are a prisoner in the Prisoner's Dilemma experiment. You and the other "
        "participants have been detained and are being interrogated separately. "
        "The prosecutor now asks: Will {name} confess to the crime, or stay silent? "
        "Remember: If all prisoners stay silent, everyone receives a moderate sentence. "
        "If you confess while others stay silent, you get a reduced sentence while they receive harsh penalties. "
        "If everyone confesses, all receive standard penalties.\n\n"
        f"{RESPONSE_FORMAT}"  # Include response format guidance
    )
    
    # Updated action spec with new terminology and call to action
    action_spec = agent_lib.choice_action_spec(
        call_to_action=call_to_action,
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
    
    # Create the scene type specification with updated terminology
    decision_scene_type = scene_lib.SceneTypeSpec(
        name=scene_type_name,
        premise={
            cfg.name: [
                f"It's time to make your decision for this round of the Prisoner's Dilemma.\n"
                f"Remember: If all prisoners stay silent, everyone receives a moderate sentence. "
                f"If you confess while others stay silent, you get a reduced sentence. "
                f"If everyone confesses, all receive standard penalties.\n\n"
                f"{RESPONSE_FORMAT}"  # Include response format in premise
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
    num_silent = sum(1 for action in joint_action.values() if action == "stay silent")
    total_players = len(joint_action)
    
    for name, action in joint_action.items():
        score = rewards[name]
        total_score = cumulative_rewards[name]
        silent_rate = num_silent / total_players * 100
        
        result[name] = (
            f"{name} chose to {action}. {num_silent} out of {total_players} "
            f"prisoners stayed silent ({silent_rate:.1f}%). {name} received {score} "
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
    ):
        """Initialize the simulation.""" 
        self._agent_module = agent_module

        self._agent_model = agent_model
        self._gm_model = gm_model
        self._embedder = embedder

        self._measurements = measurements 
        
        self._clock = game_clock.FixedIntervalClock(
            start=SETUP_TIME, step_size=TIME_STEP
        )
        
        # Define importance model for game master
        importance_model_gm = importance_function.ConstantImportanceModel()
        
        self._blank_memory_factory = blank_memories.MemoryFactory(
            model=self._gm_model,
            embedder=self._embedder,
            importance=importance_model_gm.importance,
            clock_now=self._clock.now,
        )
        
        # Get shared memories and context
        shared_memories, shared_context = get_shared_memories_and_context(self._gm_model)
        
        # Create memory factory
        self._formative_memory_factory = formative_memories.FormativeMemoryFactory(
            model=self._gm_model,
            shared_memories=shared_memories,
            blank_memory_factory_call=self._blank_memory_factory.make_blank_memory,
        )
        
        # Configure players
        player_configs = configure_agents()
        
        # Create memories for players
        tasks = {
            config.name: functools.partial(
                self._make_player_memories, config=config
            )
            for config in player_configs
        }
        self._all_memories = concurrency.run_tasks(tasks)
        
        # Create player agents
        players = []
        for player_config in player_configs:
            kwargs = dict(
                config=player_config,
                model=copy.copy(self._agent_model),
                memory=self._all_memories[player_config.name],
                clock=self._clock,
                update_time_interval=TIME_STEP,
            )
            
            player = self._agent_module.build_agent(**kwargs)
            players.append(player)
        
        self._all_players = players
        
        # Create game master
        self._environment, self._game_master_memory = (
            basic_game_master.build_game_master(
                model=self._gm_model,
                embedder=self._embedder,
                importance_model=importance_model_gm,
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
            main_player_configs=agent_configs,
        )
        
        self._payoff = SchellingPayoff(
            player_names=player_names,
            model=self._gm_model,
            memory=self._game_master_memory,
            clock_now=self._clock.now,
        )
        
        # Initialize memories
        self._init_premise_memories(
            setup_time=SETUP_TIME,
            main_player_configs=agent_configs,
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
        
        # Add response format guidance to memories
        self._game_master_memory.add(RESPONSE_FORMAT)
        for player in self._all_players:
            player.observe(RESPONSE_FORMAT)
        
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
        
        return simulation_outcome, html_results_log

