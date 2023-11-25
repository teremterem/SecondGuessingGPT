"""
SecondGuessingGPT Telegram bot that uses AgentForum under the hood.
"""
import logging
from typing import Any

from agentforum.ext.llms.openai import openai_chat_completion
from agentforum.forum import Forum, InteractionContext
from agentforum.storage import InMemoryStorage
from telegram import Update
from telegram.ext import ApplicationBuilder

from second_guessing_config import TELEGRAM_BOT_TOKEN, pl_async_openai_client

logger = logging.getLogger(__name__)

tg_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

# TODO Oleksandr: should be possible to instantiate Forum without passing any arguments (InMemoryStorage by default?)
forum = Forum(immutable_storage=InMemoryStorage())


@forum.agent
async def chat_gpt_agent(ctx: InteractionContext, **kwargs) -> None:
    """An agent that uses OpenAI ChatGPT under the hood. It sends the full chat history to the OpenAI API."""
    full_chat = await ctx.request_messages.amaterialize_full_history()
    ctx.respond(openai_chat_completion(prompt=full_chat, async_openai_client=pl_async_openai_client, **kwargs))


async def handle_telegram_update(tg_update_dict: dict[str, Any]) -> None:
    """Handle a Telegram update by calling an agent in AgentForum."""
    # pylint: disable=no-member
    tg_update = Update.de_json(tg_update_dict, tg_app.bot)
    if not (tg_update.effective_message and tg_update.effective_message.text):
        logger.debug("IGNORING UPDATE WITHOUT EFFECTIVE MESSAGE OR TEXT: %s", tg_update_dict)
        return

    assistant_responses = chat_gpt_agent.quick_call(
        tg_update.effective_message.text,
        # model="gpt-4-1106-preview",
        model="gpt-3.5-turbo-1106",
        # stream=True,
    )
    async for response in assistant_responses:
        concrete_msg = await response.amaterialize()
        await tg_update.effective_chat.send_message(concrete_msg.content)
