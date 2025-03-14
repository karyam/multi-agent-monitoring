from concordia.language_model import language_model
import goodfire
from goodfire import Variant

from dotenv import load_dotenv
import os
load_dotenv()
GOODFIRE_API_KEY = os.getenv("GOODFIRE_API_KEY")


class GoodfireModel(language_model.LanguageModel):
    client = goodfire.Client(GOODFIRE_API_KEY)

    def __init__(self, model_name: str, system_prompt: str, max_tokens: int = 500):
        self._model_name = model_name
        self._system_prompt = system_prompt
        self.model = Variant(model_name)
        self.max_tokens = max_tokens
        self.model.reset()
        
    def sample_text(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            [{"role": "system", "content": self._system_prompt},
             {"role": "user", "content": prompt}],
            model=self.model,
            stream=False,
            max_completion_tokens=self.max_tokens,
        )
        content = response.choices[0].message["content"]
        return content
    
    def sample_choice(self, prompt: str, responses: list[str], **kwargs) -> str:
        pass