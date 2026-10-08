"""TraceRelay's public Transformers API."""

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_trace_relay import TraceRelayConfig
from .modeling_trace_relay import TraceRelayCache, TraceRelayForCausalLM, TraceRelayModel

AutoConfig.register("trace_relay", TraceRelayConfig, exist_ok=True)
AutoModel.register(TraceRelayConfig, TraceRelayModel, exist_ok=True)
AutoModelForCausalLM.register(TraceRelayConfig, TraceRelayForCausalLM, exist_ok=True)

__all__ = ["TraceRelayConfig", "TraceRelayCache", "TraceRelayModel", "TraceRelayForCausalLM"]
