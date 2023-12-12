"""
This module contains the Telegram update handler that triggers AgentForum agents.
"""
import logging
from typing import Any

import telegram as tg
from agentforum.forum import ConversationTracker

from second_guessing_app.agents.forum import (
    update_msg_forum_to_tg_mappings,
    forum,
    LATEST_FORUM_HASH_IN_TG_CHAT,
    SLOW_OPENAI_MODEL,
    FORUM_HASH_TO_TG_MSG_ID,
    tg_app,
)
from second_guessing_app.agents.proxy_agent import send_forum_msg_to_telegram
from second_guessing_app.agents.second_guessing_agents import hello_agent, chat_gpt_agent, critic_agent

logger = logging.getLogger(__name__)


async def handle_telegram_update(tg_update_dict: dict[str, Any]) -> None:
    """Handle a Telegram update by calling an agent in AgentForum."""
    # pylint: disable=no-member
    tg_update = tg.Update.de_json(tg_update_dict, tg_app.bot)
    if not (tg_update.effective_message and tg_update.effective_message.text):
        logger.debug("IGNORING UPDATE WITHOUT EFFECTIVE MESSAGE OR TEXT: %s", tg_update_dict)
        return

    if tg_update.effective_message.text == "/start":
        # TODO Oleksandr: Should it be possible to update the message tree (add new messages to its branches) without
        #  calling any agents ? Because right now I am forced to create a special agent that only says "hello". I
        #  could have just created this message here, without calling any agents.
        hello_msg = await hello_agent.quick_call().amaterialize_concluding_message()

        tg_msg = await tg_update.effective_chat.send_message(hello_msg.content)
        update_msg_forum_to_tg_mappings(hello_msg.hash_key, tg_msg.message_id, tg_update.effective_chat.id)
        return

    chat_gpt_call = chat_gpt_agent.call(
        # TODO Oleksandr: optionally allow passing `branch_from` directly
        conversation=ConversationTracker(
            forum=forum,
            # TODO Oleksandr: make it possible to pass bare message hash keys as MessageType ?
            branch_from=await forum.afind_message_promise(LATEST_FORUM_HASH_IN_TG_CHAT[tg_update.effective_chat.id]),
        )
        if LATEST_FORUM_HASH_IN_TG_CHAT.get(tg_update.effective_chat.id)
        else None,
        model=SLOW_OPENAI_MODEL,
        stream=True,
    )
    chat_gpt_call.send_request(
        tg_update.effective_message.text,
        tg_message_id=tg_update.effective_message.message_id,
        tg_chat_id=tg_update.effective_chat.id,
    )
    sliced_responses = send_forum_msg_to_telegram.quick_call(
        chat_gpt_call.finish(),
        tg_chat_id=tg_update.effective_chat.id,
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO
        # TODO Oleksandr: MAJOR PROBLEM: agent calls are never awaited for if no one reads their responses - this
        #  will definitely be hard to debug/understand for anyone who is not the author of this "feature" in the
        #  framework
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO
        # TODO TODO TODO TODO TODO TODO TODO TODO TODO
    )
    async for sub_msg_promise in sliced_responses:
        sub_msg = await sub_msg_promise.amaterialize()
        update_msg_forum_to_tg_mappings(sub_msg.hash_key, sub_msg.metadata.tg_message_id, sub_msg.metadata.tg_chat_id)

    critic_responses = critic_agent.quick_call(
        # TODO Oleksandr: rename .finish() to something that implies that it is "idempotent" (because it can be called
        #  multiple times without side effects - agent processing will happen only once)
        content=chat_gpt_call.finish(),
        model=SLOW_OPENAI_MODEL,
        # TODO Oleksandr: find a way to automatically decide if streaming is needed based on who is the final
        #  receiver of the message ? how hard would that be to do that ?
        stream=True,  # TODO Oleksandr: turn it off for critic when it is not talking to the user directly anymore
    )
    sliced_responses = send_forum_msg_to_telegram.quick_call(
        critic_responses,
        tg_chat_id=tg_update.effective_chat.id,
        reply_to_tg_msg_id=FORUM_HASH_TO_TG_MSG_ID[
            # TODO TODO TODO TODO TODO TODO TODO TODO TODO
            # TODO TODO TODO TODO TODO TODO TODO TODO TODO
            # TODO TODO TODO TODO TODO TODO TODO TODO TODO
            # TODO Oleksandr: introduce agent_call.amaterialize_concluding_response() ? + ..._all_responses() ?
            #  (await chat_gpt_call.finish().amaterialize_concluding_message()).hash_key
            # TODO TODO TODO TODO TODO TODO TODO TODO TODO
            # TODO TODO TODO TODO TODO TODO TODO TODO TODO
            # TODO TODO TODO TODO TODO TODO TODO TODO TODO
            (await sliced_responses.amaterialize_concluding_message()).hash_key
        ],
    )
    async for sub_msg_promise in sliced_responses:
        sub_msg = await sub_msg_promise.amaterialize()
        update_msg_forum_to_tg_mappings(sub_msg.hash_key, sub_msg.metadata.tg_message_id, sub_msg.metadata.tg_chat_id)
