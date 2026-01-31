import pytest
from ingest_md import db_sections, parse_markdown
from langchain_text_splitters import MarkdownHeaderTextSplitter

# read the README
@pytest.fixture
def text():
    with open('../README.md', 'r', encoding='utf-8') as f:
        text = f.read()
    return text

# check the parse function
def test_sections_number(text):
    """Check the number of sections based on the table of contents in the README."""
    headers_to_split_on = [("#", "Title"), ("##", "Section")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    sections = splitter.split_text(text)
    table_of_contents = sections[1].page_content.split("\n")
    assert len(sections) == 2 + len(table_of_contents)

def test_chunks_unique_sections(text):
    """Check the parsed chunks only contain items from `db_sections`."""
    chunks = parse_markdown(text)
    sections = set([chunk.metadata.get("section") for chunk in chunks])
    for i, chunk in enumerate(chunks):
        print(i, chunk)
    assert sections == set(db_sections)