import duckdb
from langchain_core.documents import Document
from langchain_community.vectorstores import LanceDB
from langchain_openai import OpenAIEmbeddings

csv_path = "./GoodReads_100k_books.csv"
genre_to_include = "Psychology"

con = duckdb.connect()
query = f"""
    SELECT author, "desc", title, "genre", img, pages, rating, totalratings, isbn
    FROM '{csv_path}'
    WHERE instr("genre", '{genre_to_include}') > 0
"""
rows = con.execute(query).fetchall()
columns = [desc[0] for desc in con.description]

documents = [
    Document(
        page_content=row[columns.index("desc")],
        metadata={
            columns[i]: row[i] for i in range(len(columns)) if columns[i] != "desc"
        },
    )
    for row in rows
    if row[columns.index("desc")]
]

embeddings = OpenAIEmbeddings()
vector_store = LanceDB(
    uri="./db",
    embedding=embeddings,
    table_name="books",
)

vector_store.add_documents(documents)
print(f"Indexed {len(documents)} docs into LanceDB table '{vector_store._table.name}'")
