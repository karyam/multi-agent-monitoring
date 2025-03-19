## TODOs
### Management
- [ ] Decide roles and responsibilities
- [ ] Decide task management

### Planning
- [ ] Experiment design in general

### Research
- [ ] Trends in MA evaluation

### Implementation
- [ ] Check concordia experiment infrastructure that was used for the hack e.g `concordia/examples/tutorials/agent_development.ipynb` and `concordia/examples/modular`
- [ ] Use SchellingPayoffs shelling_diagram_payoffs.py  instead of PDPayoff
- [ ] Implement / use additional metrics for the `measurements` field of `Simulation`
- [ ] Integrate Goodfire, think about design carefully s.t. the codebase can be extended with custom MI methods e.g. using TransformerLens
- [ ] If we get to running models ourselves, think about infra setup, most efficient way to host models e.g vllm
- [ ] Check `concordia/concordia/factory/agent` and `concordia/concordia/factory/agent/basic_agent.py` to see whether we can rely on them instead of implementing custom agent

### Infrastructure
- [ ] Experiment infrastructure: Ray for distributed simulations, MLFlow for experiment tracking, W&B for visualisations