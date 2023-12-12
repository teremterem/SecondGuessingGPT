"""
SecondGuessingGPT Telegram bot that uses AgentForum under the hood.
"""
import asyncio
import logging

import telegram as tg
from agentforum.ext.llms.openai import openai_chat_completion
from agentforum.forum import Forum, InteractionContext
from agentforum.models import Message, Freeform
from telegram.ext import ApplicationBuilder

from second_guessing_config import TELEGRAM_BOT_TOKEN, pl_async_openai_client

SLOW_OPENAI_MODEL = "gpt-4-1106-preview"
# SLOW_OPENAI_MODEL = "gpt-3.5-turbo-1106"
FAST_OPENAI_MODEL = "gpt-3.5-turbo-1106"

FORUM_HASH_TO_TG_MSG_ID: dict[str, int] = {}
TG_MSG_ID_TO_FORUM_HASH: dict[int, str] = {}
LATEST_FORUM_HASH_IN_TG_CHAT: dict[int, str | None] = {}

tg_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

forum = Forum()

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


@forum.agent
async def send_forum_msg_to_telegram(
    ctx: InteractionContext, tg_chat_id: int = None, reply_to_tg_msg_id: int | None = None
) -> None:
    # pylint: disable=too-many-locals
    """
    Send a message from AgentForum to Telegram. Break it up into multiple Telegram messages based on the presence of
    double newlines.
    """

    async def send_tg_message(content: str) -> None:
        """
        Send a Telegram message. If `reply_to_tg_msg_id` is not None, then reply to the message with that ID and
        then set `reply_to_tg_msg_id` to None to make sure that only the first message in the series of responses
        is a reply.
        """
        nonlocal reply_to_tg_msg_id
        if reply_to_tg_msg_id is None:
            kwargs = {}
        else:
            kwargs = {"reply_to_message_id": reply_to_tg_msg_id}
            reply_to_tg_msg_id = None
        tg_message = await tg_app.bot.send_message(chat_id=tg_chat_id, text=content, **kwargs)
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO
        # TODO Oleksandr: somehow responses from this agent should NOT go after the request messages in the message
        #  tree, but as a branch that is parallel to the request messages branch ?
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO
        ctx.respond(
            content,
            tg_message_id=tg_message.message_id,
            tg_chat_id=tg_chat_id,
            openai_role="assistant",
        )

    async for msg_promise in ctx.request_messages:
        tokens_so_far: list[str] = []

        typing_task = asyncio.create_task(keep_typing(tg_chat_id))

        async for token in msg_promise:
            tokens_so_far.append(token.text)
            content_so_far = "".join(tokens_so_far)

            if content_so_far.count("```") % 2 == 1:
                # we are in the middle of a code block, let's not break it
                continue

            broken_up_content = content_so_far.rsplit("\n\n", 1)
            if len(broken_up_content) != 2:
                continue

            typing_task.cancel()

            content_left, content_right = broken_up_content

            tokens_so_far = [content_right] if content_right else []
            if content_left.strip():
                await send_tg_message(content_left)

            typing_task = asyncio.create_task(keep_typing(tg_chat_id))

        typing_task.cancel()

        remaining_content = "".join(tokens_so_far)
        if remaining_content.strip():
            await send_tg_message(remaining_content)

        # # TODO Oleksandr: what happens if I never materialize the original (full) message from openai ?
        # await msg_promise.amaterialize()


async def keep_typing(tg_chat_id: int) -> None:
    """Keep typing for up to two minutes at most."""
    for _ in range(10):
        await tg_app.bot.send_chat_action(chat_id=tg_chat_id, action=tg.constants.ChatAction.TYPING)
        await asyncio.sleep(10)


def update_msg_forum_to_tg_mappings(forum_msg_hash: str, tg_msg_id: int, tg_chat_id: int) -> None:
    """Update the mappings between forum message hashes and Telegram message IDs."""
    # TODO Oleksandr: use (tg_chat_id, tg_msg_id) tuple instead of just tg_msg_id in dictionaries below
    FORUM_HASH_TO_TG_MSG_ID[forum_msg_hash] = tg_msg_id
    TG_MSG_ID_TO_FORUM_HASH[tg_msg_id] = forum_msg_hash
    LATEST_FORUM_HASH_IN_TG_CHAT[tg_chat_id] = forum_msg_hash
