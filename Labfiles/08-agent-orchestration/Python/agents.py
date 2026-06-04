# Add references
import asyncio
import os
from dotenv import load_dotenv
from agent_framework import Agent, AgentResponse
from agent_framework.foundry import FoundryChatClient
from agent_framework.orchestrations import SequentialBuilder
from azure.identity import AzureCliCredential

load_dotenv()

# Support lab (.env) and Foundry Toolkit variable names
_PROJECT_ENDPOINT = (
    os.getenv("FOUNDRY_PROJECT_ENDPOINT")
    or os.getenv("PROJECT_ENDPOINT")
    or os.getenv("AZURE_AI_PROJECT_ENDPOINT")
)
_MODEL = (
    os.getenv("FOUNDRY_MODEL")
    or os.getenv("MODEL_DEPLOYMENT_NAME")
    or os.getenv("AZURE_AI_MODEL_DEPLOYMENT_NAME")
)


async def main():
    # Agent instructions
    summarizer_instructions="""
    Summarize the customer's feedback in one short sentence. Keep it neutral and concise.
    Example output:
    App crashes during photo upload.
    User praises dark mode feature.
    """

    classifier_instructions="""
    Classify the feedback as one of the following: Positive, Negative, or Feature request.
    """

    action_instructions="""
    Based on the summary and classification, suggest the next action in one short sentence.
    Example output:
    Escalate as a high-priority bug for the mobile team.
    Log as positive feedback to share with design and marketing.
    Log as enhancement request for product backlog.
    """

    # Create the chat client (replaces AzureAIAgentClient in Agent Framework 1.7+)
    credential = AzureCliCredential()
    chat_client = FoundryChatClient(
        credential=credential,
        project_endpoint=_PROJECT_ENDPOINT,
        model=_MODEL,
    )

    # Create agents
    summarizer = Agent(
        client=chat_client,
        instructions=summarizer_instructions,
        name="summarizer",
    )

    classifier = Agent(
        client=chat_client,
        instructions=classifier_instructions,
        name="classifier",
    )

    action = Agent(
        client=chat_client,
        instructions=action_instructions,
        name="action",
    )

    # Initialize the current feedback
    feedback="""
    I use the dashboard every day to monitor metrics, and it works well overall. 
    But when I'm working late at night, the bright screen is really harsh on my eyes. 
    If you added a dark mode option, it would make the experience much more comfortable.
    """

    # Build sequential orchestration (surface each agent's response)
    workflow = SequentialBuilder(
        participants=[summarizer, classifier, action],
        output_from="all",
    ).build()

    # Run and collect outputs
    run_result = await workflow.run(f"Customer feedback: {feedback}", stream=False)
    outputs: list[AgentResponse] = run_result.get_outputs()

    # Display outputs
    for i, response in enumerate(outputs, start=1):
        msg = response.messages[-1] if response.messages else None
        name = (
            (msg.author_name if msg else None)
            or response.agent_name
            or "assistant"
        )
        text = response.text or (msg.text if msg else "")
        print(f"{'-' * 60}\n{i:02d} [{name}]\n{text}")
    
    
if __name__ == "__main__":
    asyncio.run(main())