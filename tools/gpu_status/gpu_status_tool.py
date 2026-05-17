"""
GPU Status Tool für NAT
========================
Liefert aktuelle GPU-Utilization, Memory und Temperatur von nvidia-smi.
Funktioniert mit allen NVIDIA-GPUs, die nvidia-smi unterstützen.
"""

import subprocess
from pydantic import BaseModel, Field
from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class GPUStatusConfig(FunctionBaseConfig, name="gpu_status"):
    """
    Configuration for the GPU Status tool.

    No parameters needed — the tool always queries the local nvidia-smi.
    """
    pass


class GPUStatusInput(BaseModel):
    """Input schema for the GPU Status tool."""
    # Tool nimmt keinen Input — leerer Aufruf reicht
    pass


@register_function(config_type=GPUStatusConfig)
async def gpu_status_function(config: GPUStatusConfig, builder: Builder):
    """
    Register the gpu_status function with NAT.
    """
    async def _get_gpu_status(query: str = "") -> str:
        """
        Returns the current GPU status as a formatted string.
        Queries nvidia-smi for utilization, memory and temperature.
        """
        try:
            # Query nvidia-smi for the metrics we want
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return f"Error querying nvidia-smi: {result.stderr}"

            # Parse the CSV output
            lines = result.stdout.strip().split("\n")
            gpus = []
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 7:
                    name, util, mem_used, mem_total, temp, power_draw, power_limit = parts
                    gpus.append(
                        f"GPU {i}: {name}\n"
                        f"  - Utilization: {util}%\n"
                        f"  - Memory: {mem_used} MiB / {mem_total} MiB "
                        f"({100 * int(mem_used) / int(mem_total):.1f}%)\n"
                        f"  - Temperature: {temp}°C\n"
                        f"  - Power: {power_draw}W / {power_limit}W"
                    )

            return "\n\n".join(gpus) if gpus else "No GPUs found."

        except subprocess.TimeoutExpired:
            return "Error: nvidia-smi timed out"
        except FileNotFoundError:
            return "Error: nvidia-smi not found. Is the NVIDIA driver installed?"
        except Exception as e:
            return f"Error: {str(e)}"

    yield FunctionInfo.from_fn(
        _get_gpu_status,
        description=(
            "Returns the current status of all NVIDIA GPUs on this server "
            "including utilization in percent, memory usage, temperature in "
            "Celsius and power consumption in watts. Use this tool whenever "
            "the user asks about GPU load, GPU memory, GPU temperature or "
            "general hardware status of the inference server."
        )
    )
