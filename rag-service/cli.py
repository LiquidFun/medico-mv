import click
import requests
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

RAG_URL = os.getenv("RAG_SERVICE_URL", "http://indus:8123")


@click.group()
def cli():
    """RAG Service CLI - Document indexing and search"""
    pass


@cli.command()
@click.argument('path', type=click.Path(exists=True))
@click.option('--recursive', is_flag=True, help='Recursively index all files in directory')
def index(path: str, recursive: bool):
    """Index documents from PATH"""
    path_obj = Path(path)

    if path_obj.is_file():
        files = [path_obj]
    elif path_obj.is_dir():
        if recursive:
            files = list(path_obj.rglob('*'))
        else:
            files = list(path_obj.glob('*'))
        files = [f for f in files if f.is_file() and f.suffix.lower() in ['.pdf', '.txt', '.docx', '.doc']]
    else:
        click.echo(f"Error: {path} is not a valid file or directory")
        return

    if not files:
        click.echo("No supported files found (pdf, txt, docx)")
        return

    click.echo(f"Found {len(files)} file(s) to index")

    for file in files:
        click.echo(f"Indexing: {file.name}...", nl=False)
        try:
            response = requests.post(
                f"{RAG_URL}/index",
                json={
                    "doc_id": file.stem,  # Use filename without extension as doc_id
                    "file_path": str(file.absolute()),
                    "metadata": {"original_path": str(file)}
                },
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            click.echo(f" ✓ ({result['num_chunks']} chunks)")
        except requests.exceptions.RequestException as e:
            click.echo(f" ✗ Error: {e}")


@cli.command()
@click.argument('query')
@click.option('--top-k', default=5, help='Number of results to return')
def search(query: str, top_k: int):
    """Search indexed documents"""
    try:
        response = requests.post(
            f"{RAG_URL}/search",
            json={"query": query, "top_k": top_k},
            timeout=30
        )
        response.raise_for_status()
        result = response.json()

        if not result['chunks']:
            click.echo("No results found")
            return

        click.echo(f"\nFound {len(result['chunks'])} result(s):\n")

        for i, (chunk, score) in enumerate(zip(result['chunks'], result['scores']), 1):
            click.echo(f"--- Result {i} (score: {score:.4f}) ---")
            click.echo(f"Document: {chunk['metadata'].get('filename', chunk['doc_id'])}")
            click.echo(f"Chunk {chunk['chunk_id']}:")
            click.echo(chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text'])
            click.echo()

    except requests.exceptions.RequestException as e:
        click.echo(f"Error: {e}")


@cli.command()
def list():
    """List all indexed documents"""
    try:
        response = requests.get(f"{RAG_URL}/documents", timeout=30)
        response.raise_for_status()
        docs = response.json()

        if not docs:
            click.echo("No documents indexed")
            return

        click.echo(f"\nIndexed documents ({len(docs)}):\n")
        for doc in docs:
            click.echo(f"  • {doc['filename']}")
            click.echo(f"    ID: {doc['doc_id']}")
            click.echo(f"    Chunks: {doc['num_chunks']}")
            click.echo(f"    Uploaded: {doc.get('uploaded_at', 'N/A')}")
            click.echo()

    except requests.exceptions.RequestException as e:
        click.echo(f"Error: {e}")


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to clear all documents?')
def clear():
    """Clear all indexed documents"""
    try:
        response = requests.delete(f"{RAG_URL}/documents/all", timeout=30)
        response.raise_for_status()
        click.echo("✓ All documents cleared")
    except requests.exceptions.RequestException as e:
        click.echo(f"Error: {e}")


@cli.command()
def health():
    """Check RAG service health"""
    try:
        response = requests.get(f"{RAG_URL}/health", timeout=5)
        response.raise_for_status()
        result = response.json()
        click.echo(f"✓ Service is {result['status']}")
    except requests.exceptions.RequestException as e:
        click.echo(f"✗ Service unavailable: {e}")


if __name__ == "__main__":
    cli()
