from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings
import asyncio

# Used only for query expansion and reranking purposes
llm = init_chat_model(model_provider="openai", model="gpt-4.1-mini", temperature=0.1)
vector_store = LanceDB(uri="./db", embedding=OpenAIEmbeddings(), table_name="books")


async def transform_query(query, llm):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert at rewriting user queries for a book recommendation vector search.",
            ),
            (
                "user",
                "Rewrite the following user query to be more descriptive and optimized for a vector search.\n"
                "The goal is to find books whose descriptions match the rewritten query. Do not add any preamble.\n\n"
                f"""Original query:\n"{query}"\n\nRewritten query:""",
            ),
        ]
    )
    return await (prompt | llm | StrOutputParser()).ainvoke({"query": query})


async def fetch_books(vector_store, query, num_recommendations):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, vector_store.similarity_search, query, num_recommendations * 3
    )


async def select_top_books(llm, query, candidates, num_recommendations):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that selects the best books for a user based on their query.",
            ),
            (
                "user",
                "Based on the user's query and the provided list of books with their descriptions, "
                f"please select the top {num_recommendations} most relevant books.\n\n"
                "The selection should be based on how well the book's description aligns with the user's *original query*.\n\n"
                f"""Original User Query:\n"{query}"\n\nBook List:\n{candidates}\n\n"""
                'Please return a JSON object with a single key "selected_ids" which is a list of the integer IDs for the books you have selected. For example: [0, 3, 5]',
            ),
        ]
    )
    chain = prompt | llm | JsonOutputParser()
    return await chain.ainvoke(
        {
            "num_recommendations": num_recommendations,
            "query": query,
            "rerank_candidates": candidates,
        }
    )


@tool
async def search_books(query: str, num_recommendations: int = 5) -> list:
    """Finds and recommends the most relevant books for a given user query.

    This tool intelligently rewrites the user's query to be more effective for a
    vector search, fetches a broad set of results, and then uses an LLM to rerank
    and select the best books that match the original user intent.

    Args:
        query: Original user’s book/topic query.
        num_recommendations: How many to return (default 5).

    Returns:
        List of dicts, each describing a recommended book."""

    transformed_query = await transform_query(query, llm)
    results = await fetch_books(vector_store, transformed_query, num_recommendations)
    if not results:
        return []

    search_results_formatted = [
        {
            "title": doc.metadata.get("title"),
            "author": doc.metadata.get("author"),
            "img": doc.metadata.get("img"),
            "description": doc.page_content,
        }
        for doc in results
    ]
    rerank_candidates = "\n\n".join(
        [
            f"ID: {i}\nTitle: {doc.metadata.get('title')}\nDescription: {doc.page_content}"
            for i, doc in enumerate(results)
        ]
    )

    try:
        selection = await select_top_books(
            llm, query, rerank_candidates, num_recommendations
        )
        selected_ids = selection.get("selected_ids", [])
        if not isinstance(selected_ids, list) or not all(
            isinstance(i, int) for i in selected_ids
        ):
            return search_results_formatted[:num_recommendations]
        return [
            search_results_formatted[i]
            for i in selected_ids
            if i < len(search_results_formatted)
        ]
    except Exception:
        return search_results_formatted[:num_recommendations]
