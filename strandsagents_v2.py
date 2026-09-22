"""
Deep Agents equivalent using the CORE Strands SDK (`strands-agents`),
on OpenAI GPT via a Bedrock Application Inference Profile, with OpenTelemetry.

Verified against: strands-agents 1.56.0, strands-agents-tools, boto3 1.43.99.

    pip install strands-agents strands-agents-tools

TWO PACKAGES, ONE REPO (strands-agents/harness-sdk):
  * strands-agents   -> `from strands import Agent`        (this file)
        The core SDK. You assemble the harness yourself: pick the tools,
        wire delegation, choose a context manager. Maximum control.
  * strands-harness  -> `from strands_harness import create_harness`
        An opinionated preset built ON TOP of the core SDK. Ships builtin
        tools (read/write/edit/shell/subagent) and plugins (todos) already
        wired, behind flat kwargs.

Deep Agents mapping with the core SDK:
    read_file / write_file / edit_file  ->  file_read / file_write / editor
    execute (sandbox)                   ->  shell, python_repl, code_interpreter
    task (subagent delegation)          ->  an Agent wrapped in @tool (below),
                                            or strands.multiagent Swarm/GraphBuilder
    write_todos                         ->  journal
    SummarizationMiddleware             ->  context_manager="auto"  (core param)
    skills                              ->  strands.AgentSkills
    prompt caching                      ->  BedrockModel(cache_config=...)

Verified tool surface from the config below:
    editor, file_read, file_write, get_weather, journal, research,
    retrieve_context, think
"""

import os

import boto3
from botocore.config import Config as BotocoreConfig
from strands import Agent, tool
from strands.models import BedrockModel
from strands.telemetry import StrandsTelemetry
from strands_tools import editor, file_read, file_write, journal, think

# ---------------------------------------------------------------------------
# OpenTelemetry. The core SDK exposes the exporter setup directly -- unlike the
# harness, nothing reads OTEL_TRACES_EXPORTER for you, so call it yourself.
# Honours the standard OTEL_EXPORTER_OTLP_* env vars.
# ---------------------------------------------------------------------------
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
os.environ.setdefault("OTEL_SERVICE_NAME", "scan-harness")
StrandsTelemetry().setup_otlp_exporter()      # .setup_console_exporter() to debug

REGION = os.environ.get("AWS_REGION", "us-east-1")
AIP_ARN = os.environ["BEDROCK_AIP_ARN"]

# Corporate TLS interception: BedrockModel builds its own client and exposes no
# verify=, so set the CA bundle on the session. Never verify=False.
session = boto3.Session(region_name=REGION)
if ca := os.environ.get("CORP_CA_BUNDLE"):
    session._session.set_config_variable("ca_bundle", ca)

model = BedrockModel(
    model_id=AIP_ARN,          # AIP ARN accepted directly; no GetInferenceProfile call
    boto_session=session,      # don't also pass region_name -- that raises
    streaming=True,
    # GPT-5.x reasoning models: skip temperature/top_p.
    # additional_request_fields={"reasoning_effort": "high"},
    boto_client_config=BotocoreConfig(
        retries={"max_attempts": 3, "mode": "standard"},
        connect_timeout=10,
        read_timeout=300,
    ),
)


@tool
def get_weather(city: str) -> str:
    """Get weather for a city."""
    return f"It's always sunny in {city}!"


# ---------------------------------------------------------------------------
# Subagent: a full Agent with its OWN system prompt and its own tool subset,
# exposed to the parent as a tool. This is the "agents as tools" pattern, and
# it is what strands-harness cannot express -- its `subagent` builtin is
# generic and takes only max_depth.
# callback_handler=None keeps the subagent from printing to the parent's stream.
# ---------------------------------------------------------------------------
researcher = Agent(
    model=model,
    name="researcher",
    system_prompt="You are a careful researcher. Return a concise report.",
    tools=[get_weather],
    callback_handler=None,
)


@tool
def research(question: str) -> str:
    """Delegate a focused research question to the researcher subagent."""
    return str(researcher(question))


agent = Agent(
    model=model,
    name="orchestrator",
    system_prompt="You are a helpful assistant.",
    tools=[file_read, file_write, editor, journal, think, get_weather, research],
    context_manager="auto",       # summarization + offloading (adds retrieve_context)
    trace_attributes={            # lands on every span -- useful for cost attribution
        "app": "scan-harness",
        "inference_profile": AIP_ARN,
    },
)

if __name__ == "__main__":
    # strands_tools that touch the filesystem/shell prompt for consent by default.
    # Set BYPASS_TOOL_CONSENT=true for non-interactive runs (CI, scheduled scans).
    print(agent("What's the weather in SF?"))