"""
Energy Tracker Tool for the NVIDIA NeMo Agent Toolkit
======================================================
Reads current power consumption from Tasmota smart plugs over HTTP and
reports whether the connected appliance (washing machine, dryer, etc.)
is currently running, idle, or in low-consumption state.

A Tasmota status query returns JSON like:
    {"StatusSNS":{
        "Time":"2026-05-17T10:19:15",
        "ENERGY":{
            "Power":269, "Voltage":232, "Current":4.135,
            "Today":0.009, "Total":357.747, ...
        }
    }}

The 'Power' field (in watts) is the key indicator:
  - high value (e.g. 200+ W)  → appliance is actively running
  - moderate (5-50 W)         → warm-up, end of cycle, or standby
  - below the idle threshold  → finished or fully idle but still plugged in

Configure your devices in the YAML config:

    functions:
      energy_tracker:
        _type: energy_tracker
        idle_threshold_w: 5.0
        timeout_seconds: 5.0
        devices:
          washing_machine:
            ip: "192.168.178.20"
            aliases: ["waschmaschine", "washer", "wash"]
          dryer:
            ip: "192.168.178.52"
            aliases: ["waeschetrockner", "wäschetrockner", "trockner", "tumble dryer"]

Install with:
    cd ~/nat-playground
    source .venv/bin/activate
    uv pip install -e tools/energy_tracker
"""

import asyncio
import re

import httpx
from pydantic import BaseModel, Field

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.function import FunctionBaseConfig


class DeviceConfig(BaseModel):
    """A single Tasmota device configured in the YAML."""
    ip: str = Field(..., description="IPv4 address of the Tasmota smart plug")
    aliases: list[str] = Field(
        default_factory=list,
        description="Alternative names the LLM might use to refer to this device",
    )


class EnergyTrackerConfig(FunctionBaseConfig, name="energy_tracker"):
    """Configuration for the Energy Tracker tool."""
    devices: dict[str, DeviceConfig] = Field(
        default_factory=dict,
        description="Map of device-name -> DeviceConfig (ip + aliases)",
    )
    idle_threshold_w: float = Field(
        default=5.0,
        description="Power draw below this value (W) is considered IDLE",
    )
    timeout_seconds: float = Field(
        default=5.0,
        description="HTTP request timeout per device in seconds",
    )


# Matches IPv4 addresses anywhere in a string
_IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


async def _query_tasmota(ip: str, timeout: float) -> dict:
    """Fetch the Status 8 (energy) JSON from a single Tasmota device."""
    url = f"http://{ip}/cm?cmnd=Status%208"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


def _format_status(power_w: float, threshold: float) -> str:
    """Convert a power reading into a qualitative status string."""
    if power_w < threshold:
        return "IDLE / NOT RUNNING (appliance is finished or in standby)"
    if power_w < 50.0:
        return "LOW CONSUMPTION (warming up, standby, or end of cycle)"
    return "RUNNING (actively consuming power)"


def _format_device_report(label: str, ip: str, data: dict, threshold: float) -> str:
    """Turn a Tasmota JSON response into a human-readable status block."""
    try:
        energy = data["StatusSNS"]["ENERGY"]
        power = float(energy.get("Power", 0))
        voltage = float(energy.get("Voltage", 0))
        current = float(energy.get("Current", 0))
        today_kwh = float(energy.get("Today", 0))
        total_kwh = float(energy.get("Total", 0))

        return (
            f"--- {label} ({ip}) ---\n"
            f"Status: {_format_status(power, threshold)}\n"
            f"Current power draw: {power:.1f} W\n"
            f"Voltage: {voltage:.0f} V\n"
            f"Current: {current:.2f} A\n"
            f"Energy used today: {today_kwh:.3f} kWh\n"
            f"Energy used total: {total_kwh:.3f} kWh"
        )
    except (KeyError, ValueError, TypeError) as exc:
        return (
            f"--- {label} ({ip}) ---\n"
            f"Error parsing Tasmota response (is this really a Tasmota plug?). "
            f"Details: {type(exc).__name__}: {exc}"
        )


def _format_error(label: str, ip: str, exc: Exception, timeout: float) -> str:
    """Format a network/HTTP error as a clean message for the agent."""
    if isinstance(exc, httpx.TimeoutException):
        return (
            f"--- {label} ({ip}) ---\n"
            f"Error: device did not respond within {timeout}s. "
            f"It's likely offline, unplugged, or the IP address is wrong."
        )
    if isinstance(exc, httpx.HTTPStatusError):
        return (
            f"--- {label} ({ip}) ---\n"
            f"Error: device returned HTTP {exc.response.status_code}. "
            f"Is this actually a Tasmota device?"
        )
    if isinstance(exc, (httpx.ConnectError, httpx.RequestError)):
        return (
            f"--- {label} ({ip}) ---\n"
            f"Error: cannot reach device. Possible reasons: wrong IP address, "
            f"device offline, or not on the same network. "
            f"Details: {type(exc).__name__}"
        )
    return (
        f"--- {label} ({ip}) ---\n"
        f"Error: {type(exc).__name__}: {exc}"
    )


def _match_devices(query: str, configured: dict[str, DeviceConfig]) -> dict[str, str]:
    """
    Return the subset of configured devices that match the query.

    Matches by:
      - exact device name (case-insensitive substring of the query)
      - device name with underscores replaced by spaces
      - any of the device's aliases (case-insensitive substring)
    """
    query_lower = query.lower()
    matched: dict[str, str] = {}

    for name, device in configured.items():
        name_lower = name.lower()
        name_spaced = name_lower.replace("_", " ")

        if name_lower in query_lower or name_spaced in query_lower:
            matched[name] = device.ip
            continue

        if any(alias.lower() in query_lower for alias in device.aliases):
            matched[name] = device.ip

    return matched


@register_function(config_type=EnergyTrackerConfig)
async def energy_tracker_function(config: EnergyTrackerConfig, builder: Builder):
    """Register the energy_tracker function with NAT."""

    async def _check_energy(query: str = "") -> str:
        """
        Query Tasmota smart plug(s) for current power consumption.

        Args:
            query: Optional. Can be:
                - empty string  → queries all configured devices
                - a device name → queries that device (e.g. "washing machine")
                - an IPv4       → queries that IP directly (e.g. "192.168.178.20")
        """
        if not config.devices and not _IP_PATTERN.search(query):
            return (
                "Error: no devices are configured in the energy_tracker tool, "
                "and no IP address was provided in the query. Either configure "
                "devices in the YAML config or pass an IP address explicitly."
            )

        # Case 1: query contains an IP → use it directly (ad-hoc query)
        ip_match = _IP_PATTERN.search(query)
        if ip_match:
            ip = ip_match.group()
            label = f"Smart plug at {ip}"
            try:
                data = await _query_tasmota(ip, config.timeout_seconds)
                return _format_device_report(
                    label, ip, data, config.idle_threshold_w
                )
            except Exception as exc:
                return _format_error(label, ip, exc, config.timeout_seconds)

        # Case 2: empty or no-match query → use all configured devices
        # Case 3: query matches one or more device names/aliases → use those
        matched = _match_devices(query, config.devices) if query.strip() else {}
        devices_to_query = matched if matched else dict(
            (name, dev.ip) for name, dev in config.devices.items()
        )

        # Query selected devices in parallel for speed
        async def _query_one(name: str, ip: str) -> str:
            try:
                data = await _query_tasmota(ip, config.timeout_seconds)
                return _format_device_report(
                    name, ip, data, config.idle_threshold_w
                )
            except Exception as exc:
                return _format_error(name, ip, exc, config.timeout_seconds)

        tasks = [_query_one(name, ip) for name, ip in devices_to_query.items()]
        results = await asyncio.gather(*tasks)

        # Add a helpful preamble if the user gave a non-matching query
        if query.strip() and not matched:
            available = ", ".join(config.devices.keys())
            preamble = (
                f"Note: '{query}' did not match any configured device. "
                f"Showing status for all configured devices ({available}).\n\n"
            )
            return preamble + "\n\n".join(results)

        return "\n\n".join(results)

    yield FunctionInfo.from_fn(
        _check_energy,
        description=(
            "Reports the current power consumption (in watts) of household "
            "appliances connected to Tasmota smart plugs over HTTP. Use this "
            "tool whenever the user asks about energy consumption, whether an "
            "appliance is currently running or finished, or wants the status "
            "of a specific smart plug. Common appliances are washing machines "
            "(Waschmaschine), tumble dryers (Wäschetrockner, Trockner), and "
            "any other device on a Tasmota plug. Input options: (a) empty "
            "string '' to query ALL configured devices, (b) a device name or "
            "natural-language phrase like 'washing machine', 'Waschmaschine', "
            "'dryer', or 'Trockner' to query a specific appliance, (c) an "
            "IPv4 address like '192.168.178.20' to query an arbitrary Tasmota "
            "plug directly. Returns power draw in watts plus a qualitative "
            "status (RUNNING / LOW CONSUMPTION / IDLE). A power reading below "
            "the configured idle threshold (default 5W) means the appliance "
            "is finished or in standby; readings above 50W usually indicate "
            "active operation. Handles network errors gracefully with a clear "
            "message if a plug is unreachable."
        ),
    )
