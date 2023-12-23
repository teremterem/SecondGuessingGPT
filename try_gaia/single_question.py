"""
Try out a question from the GAIA dataset.
"""
import json
import os
from functools import partial

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

SEARCH_PDF_PROMPT = """\
Answer the following questions as best you can. You have access to the following tools:

FindPDF: Much like a search engine but finds and returns from the internet PDFs that satisfy a search query. Useful \
when the information needed to answer a question is more likely to be found in some kind of PDF document rather than \
a webpage. Input should be a search query. (NOTE: FindPDF already knows that its job is to look for PDFs, so you \
shouldn’t include the word “PDF” in your query.)

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [FindPDF]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!\
"""

EXTRACT_URL_PROMPT = """\
Your name is FindPDF. You will be provided with a SerpAPI JSON response that contains a list of search results for \
a given user query. The user is looking for a PDF document. Your job is to extract the URL that, in your opinion, \
is the most likely to contain the PDF document the user is looking for.\
"""

gpt4_completion = partial(
    openai_chat_completion,
    async_openai_client=async_openai_client,
    model="gpt-4-1106-preview",
    # model="gpt-3.5-turbo-1106",
    temperature=0,
)


@forum.agent
async def pdf_finder_agent(ctx: InteractionContext) -> None:
    """Call SerpAPI directly."""
    # TODO Oleksandr: find a prompt format that allows full chat history to be passed
    last_message = (await ctx.request_messages.amaterialize_concluding_message()).content.strip()
    prompt = [
        {
            "content": SEARCH_PDF_PROMPT,
            "role": "system",
        },
        {
            "content": f"Question: {last_message}\nThought:",
            "role": "user",
        },
    ]
    query_msg = gpt4_completion(prompt=prompt, stop="\nObservation:")

    # TODO Oleksandr: this is awkward, support StreamedMessage's own amaterialize ?
    query_msg_content = "".join([token.text async for token in query_msg])
    # get a substring that goes after "Action Input:"
    query = query_msg_content.split("Action Input:")[1]
    observation_pos = query.find("Observation:")
    if observation_pos != -1:
        query = query[:observation_pos]
    query = query.strip()

    search = GoogleSearch(
        {
            "q": query,
            "api_key": os.environ["SERPAPI_API_KEY"],
        }
    )
    organic_results = search.get_dict()["organic_results"]

    prompt = [
        {
            "content": EXTRACT_URL_PROMPT,
            "role": "system",
        },
        {
            "content": f"USER QUERY: {query}\nTHE ORIGINAL QUESTION THIS QUERY WAS DERIVED FROM: {last_message}",
            "role": "user",
        },
        {
            "content": f"SERPAPI SEARCH RESULTS: {json.dumps(organic_results)}",
            "role": "user",
        },
        {
            "content": "PLEASE ONLY RETURN A URL AND NO OTHER TEXT.\n\nURL:",
            "role": "system",
        },
    ]
    ctx.respond(gpt4_completion(prompt=prompt, stop="\nObservation:"))


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
    ctx.respond(gpt4_completion(prompt=prompt, **kwargs))


async def main() -> None:
    """Run the assistant."""

    question = (
        "What was the volume in m^3 of the fish bag that was calculated in the University of Leicester paper "
        '"Can Hiccup Supply Enough Fish to Maintain a Dragon’s Diet?"'
    )
    print("\nQUESTION:", question)

    # assistant_responses = gaia_agent.quick_call(question, stream=True)
    assistant_responses = pdf_finder_agent.quick_call(question)

    async for response in assistant_responses:
        print("\n\033[1m\033[36mGPT: ", end="", flush=True)
        async for token in response:
            print(token.text, end="", flush=True)
        print("\033[0m")
    print()
