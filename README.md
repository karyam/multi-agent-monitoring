# multi-agent-monitoring


### Tentative codebase structure

```
multi-agent-monitoring/
├── .git/                          
├── .gitignore                     
├── .env
├── requirements.txt               # Project dependencies                           
├── LICENSE                        
├── README.md                      
├── agents/                        # Agent-related code used across experiments
│   ├── __init__.py               
│   ├── basic_concordia_agent.py   # Wrapper code for each framework used
│   └── langchain_agent.py  
├── models/                        # Model-related code used across experiments
│   └── concordia_models.py        
├── prisoners_dilemma/             # Directory for Prisoner's Dilemma experiment
│   ├── environment.py             # Environment setup for Prisoner's Dilemma
│   └── experiments.ipynb          # Jupyter notebook for running experiments
├── cyber_simulation/            
│   ├── environment.py            
│   └── experiments.ipynb          
├── military_simulation/            
│   ├── environment.py             
│   └── experiments.ipynb          
└── utils/                         # Directory for utility functions
    ├── logging.py                 # Logging utility functions
    └── run_experiment.py          # Script to run experiments
```


