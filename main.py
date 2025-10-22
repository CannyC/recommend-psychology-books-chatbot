import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from tool_book_recommender_async import search_books

llm = ChatOpenAI(model="gpt-4.1", temperature=0.4)
tools = [search_books]
agent = create_agent(
    llm,
    tools,
    system_prompt="You are a helpful book recommendation assistant for psychology books only. "
    "You can use the search_books tool to find books for the user. "
    "You should show book covers to the user.",
)


@cl.on_message
async def main(message: cl.Message):
    messages = cl.user_session.get("messages")
    messages.append({"role": "user", "content": message.content})

    msg = cl.Message(content="")

    result = await agent.ainvoke({"messages": messages})

    if isinstance(result, dict) and "messages" in result:
        msg.content = extract_final_ai_content(result["messages"])
        messages.extend(result["messages"][len(messages) :])
    else:
        msg.content = str(result)

    cl.user_session.set("messages", messages)

    await msg.send()


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("messages", [])


def extract_final_ai_content(messages):
    for msg in reversed(messages):
        if (
            isinstance(msg, dict)
            and msg.get("content")
            and msg.get("type", None) in ("ai", "AIMessage")
        ):
            return msg["content"]
        if hasattr(msg, "content") and getattr(msg, "content"):
            return msg.content
    return "(No AI response found)"


@cl.password_auth_callback
async def auth_callback(username: str, password: str):
    if (username, password) == ("cansu", "cansu"):
        return cl.User(
            identifier="cansu", metadata={"role": "cansu", "provider": "credentials"}
        )
    else:
        return None
