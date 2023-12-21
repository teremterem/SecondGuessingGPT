"""
Try out a question from the GAIA dataset.
"""
import json
import os

import promptlayer
from agentforum.ext.llms.openai import openai_chat_completion
from agentforum.forum import Forum, InteractionContext
from serpapi import GoogleSearch

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

SEARCH_PROMPT = """\
Answer the following questions as best you can. You have access to the following tools:

Search: A search engine. Useful for when you need to answer questions about current events. Input should be a \
search query.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Search]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!\
"""


@forum.agent
async def pdf_finder_agent(ctx: InteractionContext) -> None:
    """Call SerpAPI directly."""
    # TODO Oleksandr: find a prompt format that allows full chat history to be passed
    last_message = (await ctx.request_messages.amaterialize_concluding_message()).content.strip()
    prompt = [
        {
            "content": SEARCH_PROMPT,
            "role": "system",
        },
        {
            "content": f"Question: {last_message}\nThought:",
            "role": "user",
        },
    ]
    query_msg = openai_chat_completion(  # TODO Oleksandr: turn this into a "partial" method
        prompt=prompt,
        async_openai_client=async_openai_client,
        model="gpt-4-1106-preview",
        # model="gpt-3.5-turbo-1106",
        temperature=0,
    )
    # TODO Oleksandr: this is awkward, support StreamedMessage's own amaterialize ?
    query_msg_content = "".join([token.text async for token in query_msg])
    # get a substring that goes after "Action Input:"
    query = query_msg_content.split("Action Input:")[1].strip()

    search = GoogleSearch(
        {
            "q": query,
            "api_key": os.environ["SERPAPI_API_KEY"],
        }
    )
    ctx.respond(json.dumps(search.get_dict()))


@forum.agent
async def gaia_agent(ctx: InteractionContext, **kwargs) -> None:
    """An agent that uses OpenAI ChatGPT under the hood. It sends the full chat history to the OpenAI API."""
    # reader = pypdf.PdfReader("../733-1496-1-SM.pdf")
    # pdf_text = "\n".join([page.extract_text() for page in reader.pages])

    # with pdfplumber.open("../733-1496-1-SM.pdf") as pdf:
    #     pdf_text = "\n".join([page.extract_text() for page in pdf.pages])

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
            "content": (
                await pdf_finder_agent.quick_call(ctx.request_messages).amaterialize_concluding_message()
            ).content,
            "role": "user",
        },
        {
            "content": "HERE GOES THE QUESTION:",
            "role": "system",
        },
        # TODO Oleksandr: should be possible to just send ctx.request_messages instead of *...
        *await ctx.request_messages.amaterialize_full_history(),
    ]
    ctx.respond(
        openai_chat_completion(  # TODO Oleksandr: turn this into a "partial" method
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
