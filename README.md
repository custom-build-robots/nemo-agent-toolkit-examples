# nemo-agent-toolkit-examples

Hands-on examples for the NVIDIA NeMo Agent Toolkit (NAT) with Ollama as local inference backend — ReAct agents, custom Python tools, and multi-agent orchestration patterns on a single GPU workstation.

## Companion Blog Posts (ai-box.eu)

This repository accompanies a three-part series on agentic AI with NAT:

1. **[NeMo Agent Toolkit on the RTX A6000 Ada — From Inference Layer to Orchestrator Layer](https://ai-box.eu/en/large-language-models-en/nemo-agent-toolkit-auf-der-rtx-a6000-ada-vom-inferenz-layer-zum-orchestrator-layer/2277/)** — Installation, Python venv setup, first ReAct agent with Ollama, and the Qwen Chinese-drift fix.
2. **[NeMo Agent Toolkit — GenAI Agent Orchestration, Local](https://ai-box.eu/en/large-language-models-en/nemo-agent-toolkit-genai-agent-orchestrierung-lokal/2288/)** — The ReAct loop in detail, tool descriptions as the primary control lever, and writing your first custom Python tool (`gpu_status` querying `nvidia-smi`).
3. **[NeMo Agent Toolkit Multi-Agent — The Supervisor Pattern, Local](https://ai-box.eu/en/news/nemo-agent-toolkit-multi-agent-supervisor-pattern-local/2309/)** — Three specialist ReAct agents under one supervisor, model-size considerations, and the `num_ctx` trap.

## Repository Layout
