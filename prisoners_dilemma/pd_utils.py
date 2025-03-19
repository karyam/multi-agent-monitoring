# Setup sentence encoder
import sentence_transformers
import collections
import concurrent.futures
import datetime

import matplotlib.pyplot as plt
import numpy as np
import sentence_transformers

from IPython import display

from concordia.agents import deprecated_agent as basic_agent
from concordia.components.agent import to_be_deprecated as components
from concordia import components as generic_components
from concordia.associative_memory import associative_memory
from concordia.associative_memory import blank_memories
from concordia.associative_memory import formative_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components import game_master as gm_components
from concordia.environment import game_master
from concordia.metrics import goal_achievement
from concordia.metrics import common_sense_morality
from concordia.metrics import opinion_of_others
from concordia.utils import html as html_lib
from concordia.utils import measurements as measurements_lib
from concordia.language_model import gpt_model
from concordia.utils import plotting


from dotenv import load_dotenv
import os

def get_sentence_encoder():
    st_model = sentence_transformers.SentenceTransformer(
        'sentence-transformers/all-mpnet-base-v2')
    embedder = lambda x: st_model.encode(x, show_progress_bar=False)
    return embedder

def get_model():
    # Load the API key from the .env file.
    load_dotenv()
    GPT_API_KEY = os.getenv('OPENAI_API_KEY')
    GPT_MODEL_NAME = 'gpt-4o' 

    if not GPT_API_KEY:
        raise ValueError('GPT_API_KEY is required.')

    model = gpt_model.GptLanguageModel(api_key=GPT_API_KEY,
                                    model_name=GPT_MODEL_NAME)
    
    return model

# model = utils.language_model_setup(
#     api_type=API_TYPE,
#     model_name=MODEL_NAME,
#     api_key=API_KEY,
#     disable_language_model=DISABLE_LANGUAGE_MODEL,
# )

def make_clock():
    time_step = datetime.timedelta(minutes=20)
    SETUP_TIME = datetime.datetime(hour=20, year=2024, month=10, day=1)

    START_TIME = datetime.datetime(hour=18, year=2024, month=10, day=2)
    clock = game_clock.MultiIntervalClock(
        start=SETUP_TIME,
        step_sizes=[time_step, datetime.timedelta(seconds=10)])
    return clock
