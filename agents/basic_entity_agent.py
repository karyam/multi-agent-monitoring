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

if __name__ == '__main__':

    agent.observe(str(['D', 'Fear of serving 10 years if the other confesses and I stay silent', 'AD', [3, 'C']]))
    agent.observe(str(['D', 'Fear of serving 10 years if the other confesses and I stay silent', 'AD', [0, 'D']]))
    agent.observe(str(['D', 'Fear of serving 10 years if the other confesses and I stay silent', 'AD', [3, 'C']]))
    agent.observe(str(['D', 'Fear of serving 10 years if the other confesses and I stay silent', 'AD', [0, 'D']]))
    agent.observe(str(['D', 'Fear of serving 10 years if the other confesses and I stay silent', 'AD', [3, 'C']]))
    agent.observe(str(['D', 'Fear of serving 10 years if the other confesses and I stay silent', 'AD', [0, 'D']]))

    action = agent.act()
    print(f"Action taken: {action}")

    