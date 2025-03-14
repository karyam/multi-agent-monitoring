from concordia.typing import entity_component
from concordia_models import GoodfireModel


class BasicConcordiaAgent(entity_component.EntityWithComponents):
    def __init__(
            self,
            agent_name: str,
            system_prompt: str,
            model_name: str,
            act_component: entity_component.ActingComponent,
            context_processor: entity_component.ContextProcessorComponent | None = None,
            context_components: Mapping[str, entity_component.ContextComponent] = (types.MappingProxyType({}))):
        super().__init__()
        self._agent_name = agent_name
        self.model = GoodfireModel(model_name, system_prompt)
        self.act_component = act_component
        self.context_processor = context_processor
        self.context_components = context_components

    def act(self, context: entity_component.Context) -> entity_component.Act:
        return self.act_component.act(context)
