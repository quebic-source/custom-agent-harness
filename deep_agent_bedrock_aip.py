"""
Deep Agents + OpenAI GPT on Amazon Bedrock via an Application Inference Profile (AIP).

Why this shape:
  * For OpenAI models on Bedrock, AIPs work ONLY with the Converse API on the
    bedrock-runtime endpoint (not Responses / Chat Completions, not bedrock-mantle).
  * So the model must be LangChain's ChatBedrockConverse (langchain-aws), not
    ChatOpenAI or the OpenAI SDK's Bedrock provider.

pip install -U deepagents langchain-aws boto3

One-time AIP creation (GPT-5.6 is cross-Region only on bedrock-runtime, so copy
from a system inference profile, not a foundation-model ARN):

  aws bedrock create-inference-profile \
    --region us-east-1 \
    --inference-profile-name deepagent-gpt56-sol \
    --model-source copyFrom=arn:aws:bedrock:us-east-1:<ACCOUNT_ID>:inference-profile/global.openai.gpt-5.6-sol \
    --tags key=project,value=deepagent key=costcenter,value=ai-platform

IAM for the caller (scope to your ARNs):
  bedrock:InvokeModel, bedrock:InvokeModelWithResponseStream  -> the AIP ARN,
      the source system profile, and the foundation-model ARNs it routes to
  bedrock:GetInferenceProfile  -> the AIP ARN (only if you omit base_model below)
"""

import os

from deepagents import create_deep_agent
from langchain_aws import ChatBedrockConverse

REGION = os.environ.get("AWS_REGION", "us-east-1")
AIP_ARN = os.environ["BEDROCK_AIP_ARN"]
# e.g. arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123xyz

llm = ChatBedrockConverse(
    model=AIP_ARN,                   # the ARN is what gets sent as modelId
    provider="openai",               # REQUIRED: langchain-aws raises if an ARN has no provider
    base_model="openai.gpt-5.6-sol", # optional; if omitted, resolved via GetInferenceProfile.
                                     # Drives streaming, tool_choice and model-profile logic.
    region_name=REGION,
    max_tokens=16_000,
    # Don't set temperature/top_p for GPT-5.x reasoning models.
    # reasoning_effort="high" is currently IGNORED for GPT-5.6 in langchain-aws
    # (issue #1262). If you need it now, pass it through raw and confirm the key
    # against the AWS model card:
    # additional_model_request_fields={"reasoning_effort": "high"},
    # Guardrails are supported on this path (Converse only):
    # guardrail_config={"guardrailIdentifier": "...", "guardrailVersion": "1"},
    request_metadata={"app": "deepagent"},  # shows up in invocation logs
)


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


research_subagent = {
    "name": "researcher",
    "description": "Does focused research on a single sub-question.",
    "system_prompt": "You are a careful researcher. Return a concise report.",
    "tools": [get_weather],
    "model": llm,  # pin subagents to the same AIP so their spend is tagged too
}

agent = create_deep_agent(
    model=llm,  # pass the instance, not a "provider:model" string
    tools=[get_weather],
    subagents=[research_subagent],
    system_prompt="You are a helpful assistant.",
)

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What's the weather in SF?"}]}
    )
    final = result["messages"][-1]
    print(final.content)
    # Confirms the call went through the AIP:
    print(final.response_metadata.get("inference_profile_id"))