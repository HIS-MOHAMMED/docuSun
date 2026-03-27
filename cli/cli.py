# cli.py
from jsonargparse import CLI
from rich.console import Console
from rich.markdown import Markdown

from engine.qa.pipeline import index_documents, query_documents

console = Console()


class DocuSunCLI:
    """DocuSun v0.2 - Local RAG Pipeline CLI"""
    
    def index(
        self,
        data_path: str = "data",
        chunk_size: int = 400,
        top_k: int = 3,
        persist_directory: str = "chroma_db",
    ):
        """Indexes documents from the data directory."""
        with console.status("[yellow]Indexing documents...[/yellow]"):
            index_documents(
                data_path=data_path,
                chunk_size=chunk_size,
                top_k=top_k,
                persist_directory=persist_directory,
            )
        console.print("[green]Indexing completed successfully.[/green]")
        
    def query(
        self,
        question: str,
        top_k: int = 3,
        persist_directory: str = "chroma_db",
    ):
        """Queries the indexed documents to answer a question."""
        with console.status("[yellow]Generating answer...[/yellow]"):
            response = query_documents(
                question=question,
                top_k=top_k,
                persist_directory=persist_directory,
            )

        console.print("[bold cyan]Answer[/bold cyan]")
        console.print(Markdown(str(response)))

def main():
    CLI(DocuSunCLI)
    
if __name__ == "__main__":
    # jsonargparse automatically converts the class methods into CLI subcommands
    main()

