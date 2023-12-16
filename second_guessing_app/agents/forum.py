"""
This module instantiates the Forum object and the Telegram app. It also defines some Telegram/AgentForum bridging
functions and global variables.
"""
import asyncio

import telegram as tg
from agentforum.forum import Forum
from telegram.ext import ApplicationBuilder

from second_guessing_config import TELEGRAM_BOT_TOKEN

SLOW_OPENAI_MODEL = "gpt-4-1106-preview"
# SLOW_OPENAI_MODEL = "gpt-3.5-turbo-1106"
FAST_OPENAI_MODEL = "gpt-3.5-turbo-1106"

FORUM_HASH_TO_TG_MSG_ID: dict[str, int] = {}
TG_MSG_ID_TO_FORUM_HASH: dict[int, str] = {}
LATEST_FORUM_HASH_IN_TG_CHAT: dict[int, str | None] = {}

tg_app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

forum = Forum()


async def keep_typing(tg_chat_id: int) -> None:
    """Keep typing for up to two minutes at most."""
    for _ in range(10):
        await tg_app.bot.send_chat_action(chat_id=tg_chat_id, action=tg.constants.ChatAction.TYPING)
        await asyncio.sleep(10)


def update_msg_forum_to_tg_mappings(forum_msg_hash: str, tg_msg_id: int, tg_chat_id: int) -> None:
    """Update the mappings between forum message hashes and Telegram message IDs."""
    # TODO Oleksandr: use (tg_chat_id, tg_msg_id) tuple instead of just tg_msg_id in dictionaries below
    # TODO Oleksandr: find a way to ALWAYS attach Telegram ids to messages as metadata ?
    FORUM_HASH_TO_TG_MSG_ID[forum_msg_hash] = tg_msg_id
    TG_MSG_ID_TO_FORUM_HASH[tg_msg_id] = forum_msg_hash
    LATEST_FORUM_HASH_IN_TG_CHAT[tg_chat_id] = forum_msg_hash
