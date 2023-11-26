"""
SecondGuessingGPT Telegram bot that uses AgentForum under the hood.
"""
import logging
from typing import Any

from agentforum.ext.llms.openai import openai_chat_completion
from agentforum.forum import Forum, InteractionContext, ConversationTracker
from agentforum.storage import InMemoryStorage
from telegram import Update
from telegram.ext import ApplicationBuilder

from second_guessing_config import TELEGRAM_BOT_TOKEN, pl_async_openai_client

logger = logging.getLogger(__name__)

tg_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

# TODO Oleksandr: it should be possible to instantiate Forum without passing any arguments (InMemoryStorage by default)
forum = Forum(immutable_storage=InMemoryStorage())

FORUM_HASH_TO_TG_MSG_ID: dict[str, int] = {}
TG_MSG_ID_TO_FORUM_HASH: dict[int, str] = {}
LATEST_FORUM_HASH_IN_TG_CHAT: dict[int, str | None] = {}


@forum.agent
async def chat_gpt_agent(ctx: InteractionContext, **kwargs) -> None:
    """An agent that uses OpenAI ChatGPT under the hood. It sends the full chat history to the OpenAI API."""
    for request_msg in await ctx.request_messages.amaterialize_all():
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
    ctx.respond("hello", openai_role="assistant")


async def handle_telegram_update(tg_update_dict: dict[str, Any]) -> None:
    """Handle a Telegram update by calling an agent in AgentForum."""
    # pylint: disable=no-member
    tg_update = Update.de_json(tg_update_dict, tg_app.bot)
    if not (tg_update.effective_message and tg_update.effective_message.text):
        logger.debug("IGNORING UPDATE WITHOUT EFFECTIVE MESSAGE OR TEXT: %s", tg_update_dict)
        return

    if tg_update.effective_message.text == "/start":
        # TODO Oleksandr: Should it be possible to update the message tree (add new messages to its branches) without
        #  calling any agents ? Because right now I am forced to create a special agent that only says "hello". I
        #  could have just created this message here, without calling any agents.
        hello_msg = await hello_agent.quick_call(None).amaterialize_concluding_message()
        # TODO Oleksandr: it should be possible not to send None at all (`content` should be an optional kwarg)

        tg_msg = await tg_update.effective_chat.send_message(hello_msg.content)
        update_msg_forum_to_tg_mappings(hello_msg.hash_key, tg_msg.message_id, tg_update.effective_chat.id)
        return

    chat_gpt_call = chat_gpt_agent.call(
        conversation=ConversationTracker(
            forum=forum,
            # TODO Oleksandr: make it possible to pass bare message hash keys as MessageType ?
            branch_from=await forum.afind_message_promise(LATEST_FORUM_HASH_IN_TG_CHAT[tg_update.effective_chat.id]),
        )
        if LATEST_FORUM_HASH_IN_TG_CHAT.get(tg_update.effective_chat.id)
        else None,
        # model="gpt-4-1106-preview",
        model="gpt-3.5-turbo-1106",
        # stream=True,
    )
    chat_gpt_call.send_request(
        tg_update.effective_message.text,
        tg_message_id=tg_update.effective_message.message_id,
        tg_chat_id=tg_update.effective_chat.id,
    )
    async for response_promise in chat_gpt_call.finish():
        response_msg = await response_promise.amaterialize()
        tg_msg = await tg_update.effective_chat.send_message(response_msg.content)
        update_msg_forum_to_tg_mappings(response_msg.hash_key, tg_msg.message_id, tg_update.effective_chat.id)


def update_msg_forum_to_tg_mappings(forum_msg_hash, tg_msg_id: int, tg_chat_id) -> None:
    """Update the mappings between forum message hashes and Telegram message IDs."""
    FORUM_HASH_TO_TG_MSG_ID[forum_msg_hash] = tg_msg_id
    TG_MSG_ID_TO_FORUM_HASH[tg_msg_id] = forum_msg_hash
    LATEST_FORUM_HASH_IN_TG_CHAT[tg_chat_id] = forum_msg_hash
