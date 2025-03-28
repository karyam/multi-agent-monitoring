from concordia.agents import entity_agent_with_logging
from concordia.typing import entity_component
from concordia.language_model import language_model
from concordia.associative_memory import formative_memories
    
def build_agent(
    config: formative_memories.AgentConfig,
    model: language_model.LanguageModel,

):
    agent_name = config.name
    memory = None
    observation = None

    act_component = None
    
    context_components = None
    
    measurements = measurements.Measurements()

    agent = entity_agent_with_logging.EntityAgentWithLogging(
        agent_name,
        act_component=act_component,
        context_components=context_components,
        component_logging=measurements,
        config=config,
    )
    
    return agent
    
    
