# nemo-agent-toolkit-examples

Hands-on examples for the NVIDIA NeMo Agent Toolkit (NAT) with Ollama as local inference backend — ReAct agents, custom Python tools, and multi-agent orchestration patterns on a single GPU workstation.

## Companion Blog Posts (ai-box.eu)

This repository accompanies a three-part series on agentic AI with NAT:

1. **[NeMo Agent Toolkit on the RTX A6000 Ada — From Inference Layer to Orchestrator Layer](https://ai-box.eu/en/large-language-models-en/nemo-agent-toolkit-auf-der-rtx-a6000-ada-vom-inferenz-layer-zum-orchestrator-layer/2277/)** — Installation, Python venv setup, first ReAct agent with Ollama, and the Qwen Chinese-drift fix.
2. **[NeMo Agent Toolkit — GenAI Agent Orchestration, Local](https://ai-box.eu/en/large-language-models-en/nemo-agent-toolkit-genai-agent-orchestrierung-lokal/2288/)** — The ReAct loop in detail, tool descriptions as the primary control lever, and writing your first custom Python tool (`gpu_status` querying `nvidia-smi`).
3. **[NeMo Agent Toolkit Multi-Agent — The Supervisor Pattern, Local](https://ai-box.eu/en/news/nemo-agent-toolkit-multi-agent-supervisor-pattern-local/2309/)** — Three specialist ReAct agents under one supervisor, model-size considerations, and the `num_ctx` trap.

## Prerequisites

- Ubuntu 24.04 LTS (or comparable Linux)
- NVIDIA GPU with current drivers - `nvidia-smi` reachable from the host
- Ollama running with the OpenAI-compatible API on port 11434
- Python 3.11, 3.12 or 3.13
- NAT installed in a Python venv - see the installation blog post above

## Quick Start

```bash
git clone https://github.com/custom-build-robots/nemo-agent-toolkit-examples.git
cd nemo-agent-toolkit-examples

# Copy configs and tools into your NAT playground
cp configs/* ~/nat-playground/configs/
cp -r tools/* ~/nat-playground/tools/

# Activate your NAT venv and install the custom tools
cd ~/nat-playground
source .venv/bin/activate
uv pip install -e tools/gpu_status
uv pip install -e tools/energy_tracker

# Pull the required Ollama models
ollama pull qwen2.5:7b-instruct      # for experiments 1, 2, 4
ollama pull qwen3.6:27b              # for experiment 3 (multi-agent)

# Run an experiment
cd ~/nat-playground/configs
nat run --config_file experiment2_gpu_agent.yml \
  --input "What is the current GPU utilization?"
```

## Tested On

- **Workstation:** NVIDIA RTX A6000 Ada (48 GB VRAM)
- **OS:** Ubuntu 24.04 LTS
- **NAT version:** 1.6
- **Ollama version:** 0.5+

## License

[MIT](LICENSE)

## Author

[Ingmar Stapel](https://github.com/custom-build-robots) — maker, AI infrastructure tinkerer, author of [ai-box.eu](https://ai-box.eu/).
