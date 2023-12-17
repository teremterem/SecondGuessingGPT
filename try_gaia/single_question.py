# pylint: disable=wrong-import-position
"""
Try out a question from the GAIA dataset.
"""
import asyncio

# noinspection PyUnresolvedReferences
import readline  # pylint: disable=unused-import
import warnings

from dotenv import load_dotenv

load_dotenv()

import promptlayer

# TODO Oleksandr: get rid of this warning suppression when PromptLayer doesn't produce "Expected Choice but got dict"
#  warning anymore
warnings.filterwarnings("ignore", module="pydantic")

from agentforum.ext.llms.openai import openai_chat_completion
from agentforum.forum import Forum, InteractionContext

forum = Forum()
async_openai_client = promptlayer.openai.AsyncOpenAI()

GAIA_SYSTEM_PROMPT = """\
You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the \
following template: FINAL ANSWER: [YOUR FINAL ANSWER].
YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
If you are asked for a number, don’t use comma to write your number neither use units such as $ or percent sign \
unless specified otherwise.
If you are asked for a string, don’t use articles, neither abbreviations (e.g. for cities), and write the digits in \
plain text unless specified otherwise.
If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the \
list is a number or a string.\
"""


@forum.agent
async def gaia_agent(ctx: InteractionContext, **kwargs) -> None:
    """An agent that uses OpenAI ChatGPT under the hood. It sends the full chat history to the OpenAI API."""
    full_chat = await ctx.request_messages.amaterialize_full_history()
    prompt = [
        {
            "content": GAIA_SYSTEM_PROMPT,
            "role": "system",
        },
        *full_chat,
    ]
    ctx.respond(openai_chat_completion(prompt=prompt, async_openai_client=async_openai_client, **kwargs))


async def main() -> None:
    """Run the assistant."""
    question = (
        "What was the volume in m^3 of the fish bag that was calculated in the University of Leicester paper "
        '"Can Hiccup Supply Enough Fish to Maintain a Dragon’s Diet?"'
    )
    print("\nQUESTION:", question)

    assistant_responses = gaia_agent.quick_call(
        question,
        model="gpt-4-1106-preview",
        # model="gpt-3.5-turbo-1106",
        temperature=0,
        stream=True,
    )

    async for response in assistant_responses:
        print("\n\033[1m\033[36mGPT: ", end="", flush=True)
        async for token in response:
            print(token.text, end="", flush=True)
        print("\033[0m")
    print()


if __name__ == "__main__":
    asyncio.run(main())
