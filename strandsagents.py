"""
Strands Harness SDK + OpenAI GPT on Amazon Bedrock via an Application Inference
Profile (AIP), with OpenTelemetry.

Verified against: strands-harness 0.1.1, strands-agents 1.56.0, boto3 1.43.99.

    pip install strands-harness

Equivalent to the Deep Agents setup, but the harness ships these as *builtin
tools* rather than middleware:

    Deep Agents                     Strands Harness
    ---------------------------     ----------------------------------
    read_file / write_file / edit   read / write / edit  (builtin_tools)
    task (subagent delegation)      subagent             (builtin_tools)
    write_todos (TodoListMiddleware) todo_write          (builtin_plugins)
    SummarizationMiddleware         context_manager="auto"
    context offloading              retrieve_offloaded_content
    sandbox execute                 shell                (builtin_tools)
    interpreter / eval              programmatic_tool_caller
    AGENTS.md memory                memory=True  -> ./.agent/memory
    skills                          skills=True  -> ./.agent/skills

Verified tool surface from the config below:
    edit, get_weather, programmatic_tool_caller, read, retrieve_context,
    retrieve_offloaded_content, strands_manage_background_task, subagent,
    todo_write, write
"""

import os

import boto3
from botocore.config import Config as BotocoreConfig
from strands import tool
from strands.models import BedrockModel
from strands_harness import create_harness

# ---------------------------------------------------------------------------
# OpenTelemetry. The harness follows the standard OTEL env convention and wires
# its own exporter -- no instrumentor package, no manual TracerProvider.
#   OTEL_TRACES_EXPORTER = otlp | console
# ---------------------------------------------------------------------------
os.environ.setdefault("OTEL_TRACES_EXPORTER", "otlp")
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
os.environ.setdefault("OTEL_SERVICE_NAME", "scan-harness")

REGION = os.environ.get("AWS_REGION", "us-east-1")
AIP_ARN = os.environ["BEDROCK_AIP_ARN"]
# arn:aws:bedrock:us-east-1:<acct>:application-inference-profile/<id>

# ---------------------------------------------------------------------------
# Corporate TLS interception (Zscaler/Netskope). BedrockModel builds its own
# boto3 client and exposes no verify= parameter, so set the CA bundle on the
# session. All three of these were verified to reach the client's verify:
#   1. this one -- session config variable, no env vars needed
#   2. AWS_CA_BUNDLE env var
#   3. model.client = boto3.client("bedrock-runtime", verify=CA) after init
# Never verify=False.
# ---------------------------------------------------------------------------
session = boto3.Session()
if ca := os.environ.get("CORP_CA_BUNDLE"):
    session._session.set_config_variable("ca_bundle", ca)

model = BedrockModel(
    model_id=AIP_ARN,          # AIP ARNs are accepted directly as model_id
    boto_session=session,      # do NOT also pass region_name -- it raises
    streaming=True,
    # GPT-5.x reasoning models: skip temperature/top_p.
    # `effort=` on create_harness is IGNORED for a pre-built Model instance
    # (it warns), so configure reasoning here instead:
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


agent = create_harness(
    model=model,
    instructions="You are a helpful assistant.",
    tools=[get_weather],
    builtin_tools={
        "read": True, "write": True, "edit": True,
        "subagent": {"max_depth": 2},   # delegation depth; default is 2
        "shell": False,                 # off by default here -- opt in deliberately
        "web_search": False, "web_fetch": False,
    },
    builtin_plugins=["todos"],          # todo_write, the write_todos equivalent
    context_manager="auto",             # summarization + offloading
    # Caching is OFF: with an ARN model_id, auto cache-strategy detection cannot
    # see the underlying model. Bedrock's Converse path for OpenAI models has no
    # prompt caching anyway (it's Responses-API only).
    caching=False,
    session=False, memory=False, skills=False,   # flip on to persist under ./.agent/
)

if __name__ == "__main__":
    result = agent("What's the weather in SF?")
    print(result)