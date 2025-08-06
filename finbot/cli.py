"""
FinBot Command Line Interface

A CLI application for interacting with the FinBot financial assistant.
Supports document ingestion and interactive question-answering.
"""

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

from rich import print
from rich.prompt import Prompt
from finbot.ingestion.ingest import ingest
from finbot.embedding.embedder import embed
from finbot.retriever.similarity import retrieve_similar
from finbot.prompt.formatter import build_prompt
from finbot.llm import get_llm
from finbot.config import TOP_K, EMBED_MODEL, MAX_RESPONSE_TOKENS


def interactive():
    """Run interactive Q&A session with the financial assistant."""
    llm = get_llm()
    print("[bold green]FinBot ready › Ask questions about Canadian finance[/bold green]")
    
    while True:
        try:
            query = Prompt.ask("\nQuery (or 'exit')")
            if query.lower() in {"exit", "quit"}:
                break
                
            query_embedding = embed([query])[0]
            similar_chunks = retrieve_similar(query_embedding, TOP_K)
            
            if not similar_chunks:
                print("[yellow]No relevant documents found for your question.[/yellow]")
                continue
                
            prompt = build_prompt(query, similar_chunks)
            
            print("\n[bold blue]Answer:[/bold blue]")
            response_tokens = []
            token_count = 0
            max_response_tokens = MAX_RESPONSE_TOKENS
            
            for token in llm.stream(prompt):
                print(token, end="", flush=True)
                response_tokens.append(token)
                token_count += len(token.split())
                
                # Prevent excessively long responses
                if token_count > max_response_tokens:
                    print("\n[dim]...(response truncated)[/dim]")
                    break
            
            # Display source attribution
            if similar_chunks:
                source = similar_chunks[0].get('source', 'Unknown')
                print(f"\n\n[dim]Source: {source}[/dim]")
            print("\n" + "-" * 50)
            
        except KeyboardInterrupt:
            print("\n[yellow]Interrupted by user[/yellow]")
            break
        except Exception as e:
            print(f"[red]Error:[/red] {e}")


def main():
    """Main entry point for the CLI application."""
    parser = argparse.ArgumentParser(
        description="FinBot - Canadian Financial Assistant",
        epilog="For more information, visit: https://github.com/your-username/finbot"
    )
    parser.add_argument(
        "--ingest", 
        action="store_true",
        help="Ingest documents from data/raw/ directory"
    )
    
    args = parser.parse_args()
    
    if args.ingest:
        ingest()
    else:
        interactive()


if __name__ == "__main__":
    main()
