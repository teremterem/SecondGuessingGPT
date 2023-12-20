"""
Try out a question from the GAIA dataset.
"""
import json
import os
from pprint import pprint

import promptlayer
from agentforum.ext.llms.openai import openai_chat_completion
from agentforum.forum import Forum, InteractionContext
from serpapi import GoogleSearch

from try_gaia.captured_serpapi_result import CAPTURED_SERPAPI_RESULT

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
async def pdf_finder_agent(ctx: InteractionContext) -> None:
    """Call SerpAPI directly."""
    query = (await ctx.request_messages.amaterialize_concluding_message()).content
    search = GoogleSearch(
        {
            "q": query,
            "api_key": os.environ["SERPAPI_API_KEY"],
        }
    )
    result = search.get_dict()
    pprint(result)


@forum.agent
async def gaia_agent(ctx: InteractionContext, **kwargs) -> None:
    """An agent that uses OpenAI ChatGPT under the hood. It sends the full chat history to the OpenAI API."""
    # reader = pypdf.PdfReader("../733-1496-1-SM.pdf")
    # pdf_text = "\n".join([page.extract_text() for page in reader.pages])

    # with pdfplumber.open("../733-1496-1-SM.pdf") as pdf:
    #     pdf_text = "\n".join([page.extract_text() for page in pdf.pages])

    full_chat = await ctx.request_messages.amaterialize_full_history()
    prompt = [
        {
            "content": GAIA_SYSTEM_PROMPT,
            "role": "system",
        },
        {
            # "content": "In order to answer the question use the following content of a PDF document:",
            "content": "In order to answer the question use the following data:",
            "role": "system",
        },
        {
            "content": json.dumps(CAPTURED_SERPAPI_RESULT),
            "role": "user",
        },
        {
            "content": "HERE GOES THE QUESTION:",
            "role": "system",
        },
        *full_chat,
    ]
    ctx.respond(
        openai_chat_completion(
            prompt=prompt,
            async_openai_client=async_openai_client,
            model="gpt-4-1106-preview",
            # model="gpt-3.5-turbo-1106",
            temperature=0,
            **kwargs,
        )
    )


async def main() -> None:
    """Run the assistant."""

    question = (
        "What was the volume in m^3 of the fish bag that was calculated in the University of Leicester paper "
        '"Can Hiccup Supply Enough Fish to Maintain a Dragon’s Diet?"'
    )
    print("\nQUESTION:", question)

    assistant_responses = gaia_agent.quick_call(question, stream=True)
    # assistant_responses = pdf_finder_agent.quick_call(question)

    async for response in assistant_responses:
        print("\n\033[1m\033[36mGPT: ", end="", flush=True)
        async for token in response:
            print(token.text, end="", flush=True)
        print("\033[0m")
    print()
