from concordia.typing import entity_component
from concordia_models import language_model

#TODO: might want to refactor components into a separate file to be used by other agents?

class ObservingComponent(entity_component.ObservingComponent):
    def __init__(self, model: language_model.LanguageModel):
        self.model = model

    def get_observation(
        self, context: entity_component.Context) -> entity_component.Observation:
        return self.model.get_observation(context)


class ActingComponent(entity_component.ActingComponent):
    def __init__(self, model: language_model.LanguageModel):
        self.model = model

    def get_action_attempt(
        self, 
        context: entity_component.Context, 
        action_spec: entity_component.ActionSpec
    ) -> str:
        return self.model.get_action_attempt(context, action_spec)

#TODO: How to integrate subsequenced edits into experiment design?
#TODO: Define list of properties associated with cooperation-relevant behaviour.

class BasicConcordiaAgent(entity_component.EntityWithComponents):
    def __init__(
        self,
        agent_name: str,
        system_prompt: str,
        model_name: str,
        act_component: entity_component.ActingComponent,
        context_processor: entity_component.ContextProcessorComponent | None = None,
        context_components: Mapping[str, entity_component.ContextComponent] = types.MappingProxyType({})
    ):
        super().__init__()
        self._agent_name = agent_name
        self.model = GoodfireModel(model_name, system_prompt)
        self.act_component = act_component
        self.context_processor = context_processor
        self.context_components = context_components

    def act(self, context: entity_component.Context) -> entity_component.Act:
        return self.act_component.act(context)
    
    def 
    
    
    
