# cli.py
from typing import Annotated

from jsonargparse import CLI, ActionYesNo
from rich.console import Console
from rich.markdown import Markdown

from engine.qa.pipeline import index_documents, query_documents

console = Console()


class DocuSunCLI:
    """DocuSun v0.2 - Local RAG Pipeline CLI"""

    def _log(self, kind: str, message: str, value=None) -> None:
        if kind == "step":
            console.print(f"[yellow]• {message}[/yellow] [green]OK[/green]")
            return
        if kind == "kv":
            console.print(f"[magenta]{message}:[/magenta] {value}")
            return
        if kind == "list":
            console.print(f"[cyan]{message}[/cyan]")
            for item in value or []:
                console.print(f"[yellow]  •[/yellow] {item}")
            return
        if kind == "chunks":
            console.print(f"[cyan]{message}[/cyan]")
            for item in value or []:
                console.print(f"[yellow]  •[/yellow] {item}")
            return
        console.print(f"[yellow]• {message}[/yellow]")
    
    def index(
        self,
        data_path: str = "data",
        chunk_size: int = 400,
        top_k: int = 3,
        persist_directory: str | None = None,
        verbose: Annotated[bool, ActionYesNo] = False,
        report: str | None = None,
    ):
        """Indexes documents from the data directory."""
        with console.status("[yellow]Indexing documents...[/yellow]"):
            index_documents(
                data_path=data_path,
                chunk_size=chunk_size,
                top_k=top_k,
                persist_directory=persist_directory,
                log_fn=self._log if verbose else None,
                report_path=report,
            )
        console.print("[green]Indexing completed successfully.[/green]")
        
    def query(
        self,
        question: str,
        top_k: int = 3,
        persist_directory: str | None = None,
        verbose: Annotated[bool, ActionYesNo] = False,
        report: str | None = None,
    ):
        """Queries the indexed documents to answer a question."""
        with console.status("[yellow]Generating answer...[/yellow]"):
            response = query_documents(
                question=question,
                top_k=top_k,
                persist_directory=persist_directory,
                log_fn=self._log if verbose else None,
                report_path=report,
            )

        console.print("[bold cyan]Answer[/bold cyan]")
        console.print(Markdown(str(response)))

def main():
    CLI(DocuSunCLI)
    
if __name__ == "__main__":
    # jsonargparse automatically converts the class methods into CLI subcommands
    main()

