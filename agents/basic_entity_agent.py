import sentence_transformers

from concordia import typing
from concordia.typing import entity

from concordia.agents import entity_agent
from concordia.associative_memory import associative_memory

from concordia.components.agent import action_spec_ignored
from concordia.components.agent import memory_component
from concordia.memory_bank import legacy_associative_memory
from concordia.typing import entity_component

from concordia.language_model import gpt_model
from concordia.language_model import language_model

from dotenv import load_dotenv
import os
from prisoners_dilemma.pd_utils import get_model, get_sentence_encoder

model = get_model()
embedder = get_sentence_encoder()


class Observe(entity_component.ContextComponent):
  
  # FULLY OBSERVED: observation format: [action, reason, strategy, [opponent_action, score]

  def pre_observe(self, observation: str) -> None:
    self.get_entity().get_component('memory').add(observation, {})


class RecentMemories(entity_component.ContextComponent):

  def pre_act(self, action_spec) -> None:
    recent_memories_list = self.get_entity().get_component('memory').retrieve(
        query='',  # Don't need a query to retrieve recent memories.
        limit=5,
        scoring_fn=legacy_associative_memory.RetrieveRecent(),
    )
    recent_memories = " ".join(memory.text for memory in recent_memories_list)
    print(f"*****\nDEBUG: Recent memories:\n  {recent_memories}\n*****")
    return recent_memories


class SimpleActing(entity_component.ActingComponent):

  def __init__(self, model: language_model.LanguageModel):
    self._model = model

  def get_action_attempt(
      self,
      contexts,
      action_spec,
  ) -> str:
    # Put context from all components into a string, one component per line.
    context_for_action = "\n".join(
        f"{name}: {context}" for name, context in contexts.items()
    )
    print(f"*****\nDEBUG:\n  context_for_action:\n{context_for_action}\n*****")
    # Ask the LLM to suggest an action attempt.
    call_to_action = action_spec.call_to_action.format(
        name=self.get_entity().name, timedelta='2 minutes')
    sampled_text = self._model.sample_text(
        f"{context_for_action}\n\n{call_to_action}\n",
    )
    return sampled_text

class Generate_PD_Action(entity_component.ActingComponent):
  
    def __init__(self, model: language_model.LanguageModel):
        self._model = model
        
        self.game_history = []
        self.strategy = ""
        
        # Define as an instance variable
        self.call_to_action_template = (
        'The prosecutor now asks: Will you confess to the crime, or stay silent? '
        'Game history has fields: move_taken, move_reason, strategy, pay_off, opponent_move. '
        'Your opponent has played the following moves so far: {perceived_history} '
        'Make your best guess. Remember, the other prisoner is making the same decision without knowing yours. '
        'Respond ONLY in this format: {{"move": "C" or "D", "reason": "<brief explanation>"}} '
        'C means you **stay silent**; D means you **confess**.'
        )

        # self.client = goodfire.Client(GOODFIRE_API_KEY)
        # self.variant = goodfire.Variant("meta-llama/Llama-3.3-70B-Instruct")

    def get_action_attempt(
      self,
      contexts,
      action_spec,
    ) -> str:
        
        perceived_history = self.extract_history_from_context(contexts)

        # Put context from all components into a string, one component per line.
        context_for_action = "\n".join( f"{name}: {context}" for name, context in contexts.items() )
        print(f"*****\nDEBUG:\n  context_for_action:\n{context_for_action}\n*****")

        # Ask the LLM to suggest an action attempt.
        formatted_call_to_action = self.call_to_action_template.format(
            perceived_history=perceived_history)
        sampled_text = self._model.sample_text(
            f"{context_for_action}\n\n{formatted_call_to_action}\n",
        )
        return sampled_text

    def extract_history_from_context(self, contexts):
        # Put context from all components into a string, one component per line.
        context_for_action = "\n".join(
            f"{name}: {context}" for name, context in contexts.items()
        )

        return str(context_for_action)


raw_memory = legacy_associative_memory.AssociativeMemoryBank(
    associative_memory.AssociativeMemory(embedder))

# Let's create an agent with the above components.
agent = entity_agent.EntityAgent(
    'Alice',
    act_component=Generate_PD_Action(model),
    context_components={
        'observation': Observe(),
        'recent_memories': RecentMemories(),
        'memory': memory_component.MemoryComponent(raw_memory),
    })

# ...existing code...

class SimplePDEnvironment:
    """A simple Prisoner's Dilemma environment that works with EntityAgents."""
    
    def __init__(self, agent1, agent2, max_rounds=10):
        """Initialize with two agents."""
        self.agents = {
            agent1.name: agent1,
            agent2.name: agent2
        }
        self.agent_names = [agent1.name, agent2.name]
        self.max_rounds = max_rounds
        self.current_round = 0
        self.scores = {name: 0 for name in self.agent_names}
        self.history = {name: [] for name in self.agent_names}
        
        # Classic prisoner's dilemma payoffs (R,S,T,P format)
        self.payoff_matrix = {
            ("C", "C"): (3, 3),  # Both cooperate
            ("C", "D"): (0, 5),  # Player 1 coop, Player 2 defect
            ("D", "C"): (5, 0),  # Player 1 defect, Player 2 coop
            ("D", "D"): (1, 1),  # Both defect
        }
    
    def _parse_action(self, action_str):
        """Parse agent's action response to extract move and reason."""
        import re
        
        # Try to extract JSON format first
        json_match = re.search(r'\{.*"move"\s*:\s*"([CD])".*"reason"\s*:\s*"(.+?)".*\}', action_str, re.DOTALL)
        if json_match:
            return json_match.group(1), json_match.group(2)
        
        # Fallback to simple pattern matching
        move_match = re.search(r'([CD])', action_str)
        if move_match:
            move = move_match.group(1)
            # Try to extract a reason if present
            reason_match = re.search(r'reason:?\s*(.+?)(?:\.|$)', action_str, re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "No explicit reason given"
            return move, reason
        
        # Default if no pattern matches
        return "D", "Failed to parse action"
    
    def step(self):
        """Run one round of the game."""
        if self.current_round >= self.max_rounds:
            return False  # Game is over
        
        self.current_round += 1
        print(f"\n===== ROUND {self.current_round} =====")
        
        # Get actions from agents
        actions = {}
        reasons = {}
        
        for name in self.agent_names:
            # Format previous history for opponent
            opponent = [opp for opp in self.agent_names if opp != name][0]
            opponent_history = []
            for i in range(len(self.history[opponent])):
                # Format as [move, reason, strategy?, [score, opponent_move]]
                opp_move = self.history[opponent][i]
                my_move = self.history[name][i] if i < len(self.history[name]) else None
                
                if my_move:
                    opponent_history.append(
                        f"Round {i+1}: Opponent played {opp_move[0]} with reason: '{opp_move[1]}'. " + 
                        f"You played {my_move[0]} and received {my_move[2]} points."
                    )
            
            # Let the agent observe the game state
            game_state = (
                f"Game status: Round {self.current_round}/{self.max_rounds}\n"
                f"Your score: {self.scores[name]}\n"
                f"Opponent's score: {self.scores[opponent]}\n"
                f"Previous rounds: {'; '.join(opponent_history) if opponent_history else 'None'}"
            )
            
            self.agents[name].observe(game_state)
            
            # Get agent's action
            action_text = self.agents[name].act()
            print(f"{name}'s action: {action_text}")
            
            move, reason = self._parse_action(action_text)
            actions[name] = move
            reasons[name] = reason
            print(f"{name} chose: {move} - Reason: {reason}")
        
        # Process payoffs
        a1, a2 = self.agent_names
        move_pair = (actions[a1], actions[a2])
        payoffs = self.payoff_matrix.get(move_pair)
        
        # Update scores
        self.scores[a1] += payoffs[0]
        self.scores[a2] += payoffs[1]
        
        # Record history
        self.history[a1].append((actions[a1], reasons[a1], payoffs[0]))
        self.history[a2].append((actions[a2], reasons[a2], payoffs[1]))
        
        # Report results
        print(f"\nRound {self.current_round} results:")
        print(f"{a1}: {actions[a1]} → +{payoffs[0]} points (Total: {self.scores[a1]})")
        print(f"{a2}: {actions[a2]} → +{payoffs[1]} points (Total: {self.scores[a2]})")
        
        return True  # Game continues
    
    def run_simulation(self):
        """Run the full simulation for max_rounds."""
        while self.step():
            pass
        
        # Final results
        print("\n===== GAME OVER =====")
        print("Final scores:")
        for name in self.agent_names:
            print(f"{name}: {self.scores[name]} points")
        
        # Determine winner
        max_score = max(self.scores.values())
        winners = [name for name, score in self.scores.items() if score == max_score]
        
        if len(winners) > 1:
            print("Game ended in a tie!")
        else:
            print(f"Winner: {winners[0]}!")
        
        return self.scores, self.history


if __name__ == '__main__':
    # Create two agents with different personalities
    
    # First agent is already defined above as 'agent' (Alice)
    
    # Create the second agent (Bob)
    bob_memory = legacy_associative_memory.AssociativeMemoryBank(
        associative_memory.AssociativeMemory(embedder))
    
    # Add some memories to Bob
    bob_memory.add("My name is Bob.", {})
    bob_memory.add("My goal: I want to minimize my jail time while being fair.", {})
    bob_memory.add("My traits: I tend to be cooperative and trusting.", {})
    
    bob = entity_agent.EntityAgent(
        'Bob',
        act_component=Generate_PD_Action(model),
        context_components={
            'observation': Observe(),
            'recent_memories': RecentMemories(),
            'memory': memory_component.MemoryComponent(bob_memory),
        })
    
    # Add formative memories to Alice (already created as 'agent')
    raw_memory.add("My name is Alice.", {})
    raw_memory.add("My goal: I want to get the shortest sentence possible, no matter what.", {})
    raw_memory.add("My traits: I'm strategic and self-interested.", {})
    
    # Initialize the environment with both agents
    pd_env = SimplePDEnvironment(agent, bob, max_rounds=10)
    
    # Run the simulation
    final_scores, history = pd_env.run_simulation()
    
    # Summary analysis
    print("\n===== STRATEGY ANALYSIS =====")
    for name in pd_env.agent_names:
        cooperation_rate = sum(1 for move, _, _ in history[name] if move == "C") / len(history[name])
        print(f"{name}'s cooperation rate: {cooperation_rate:.1%}")
# if __name__ == '__main__':

#     agent.observe(str(['D', 'Fear of serving 10 years if the other confesses and I stay silent', 'AD', [3, 'C']]))
#     agent.observe(str(['D', 'Fear of serving 10 years if the other confesses and I stay silent', 'AD', [0, 'D']]))
#     agent.observe(str(['D', 'Fear of serving 10 years if the other confesses and I stay silent', 'AD', [3, 'C']]))
#     agent.observe(str(['D', 'Fear of serving 10 years if the other confesses and I stay silent', 'AD', [0, 'D']]))
#     agent.observe(str(['D', 'Fear of serving 10 years if the other confesses and I stay silent', 'AD', [3, 'C']]))
#     agent.observe(str(['D', 'Fear of serving 10 years if the other confesses and I stay silent', 'AD', [0, 'D']]))

#     action = agent.act()
#     print(f"Action taken: {action}")

    