"""
SecondGuessingGPT Telegram bot that uses AgentForum under the hood.
"""
import logging

from agentforum.ext.llms.openai import openai_chat_completion
from agentforum.forum import InteractionContext
from agentforum.models import Message, Freeform

from second_guessing_app.agents.forum import forum, update_msg_forum_to_tg_mappings
from second_guessing_config import pl_async_openai_client

logger = logging.getLogger(__name__)


@forum.agent
async def critic_agent(ctx: InteractionContext, **kwargs) -> None:
    """An agent that uses OpenAI ChatGPT to criticize the way an assistant interacts with a user."""
    # TODO Oleksandr: Introduce some kind of DetachedMessage that doesn't get "forwarded" when it becomes part of a
    #  message tree ? Or maybe just start supporting plain dictionaries in openai_chat_completion() ?
    system_msg = Message(
        content="You are a critic. Your job is to criticize the assistant. Pick on the way it talks.",
        sender_alias="Critic",  # TODO Oleksandr: I shouldn't have to worry about this field
        # TODO Oleksandr: all the unrecognized fields passed to Message directly should be treated as metadata
        metadata=Freeform(openai_role="system"),
    )
    full_chat = await ctx.request_messages.amaterialize_full_history()
    ctx.respond(
        openai_chat_completion(prompt=[system_msg, *full_chat], async_openai_client=pl_async_openai_client, **kwargs)
    )


@forum.agent
async def chat_gpt_agent(ctx: InteractionContext, **kwargs) -> None:
    """An agent that uses OpenAI ChatGPT under the hood. It sends the full chat history to the OpenAI API."""
    for request_msg in await ctx.request_messages.amaterialize_as_list():
        # TODO Oleksandr: Support some sort of pre-/post-materialization hooks to decouple message to platform mapping
        #  from the agent logic ? When to set those hooks, though ? And what should be their scope ?
        update_msg_forum_to_tg_mappings(
            request_msg.hash_key, request_msg.metadata.tg_message_id, request_msg.metadata.tg_chat_id
        )

    full_chat = await ctx.request_messages.amaterialize_full_history()
    ctx.respond(openai_chat_completion(prompt=full_chat, async_openai_client=pl_async_openai_client, **kwargs))


@forum.agent
async def hello_agent(ctx: InteractionContext) -> None:
    """An agent that sends a "hello" message."""
    ctx.respond("Hello! How can I assist you today?", openai_role="assistant")
