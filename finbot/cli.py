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
from finbot.retriever.similarity import retrieve_similar, rerank_chunks
from finbot.prompt.formatter import build_prompt
from finbot.llm import get_llm
from finbot.config import TOP_K, MAX_RESPONSE_TOKENS, LLM_BACKEND
import time


def interactive():
    """Run interactive Q&A session with the financial assistant."""
    llm = get_llm()
    print("[bold green]FinBot ready > Ask questions about Canadian finance[/bold green]")
    
    while True:
        try:
            query = Prompt.ask("\nQuery (or 'exit')")
            if query.lower() in {"exit", "quit"}:
                break
                
            # Embedding
            query_embedding = embed([query])[0]
            # Retrieval (vector search)
            start_time = time.time()
            similar_chunks = retrieve_similar(query_embedding, TOP_K)
            # Optional reranking for cloud backends
            if LLM_BACKEND in {"openai", "hf_hub"}:
                similar_chunks = rerank_chunks(query, similar_chunks, TOP_K)
            latency_ms = (time.time() - start_time) * 1000
            print(f"[dim]Retrieval+Rerank latency: {latency_ms:.0f} ms[/dim]")
            
            if not similar_chunks:
                print("[yellow]No relevant documents found for your question.[/yellow]")
                continue
                
            prompt = build_prompt(query, similar_chunks)
            
            print("\n[bold blue]Answer:[/bold blue]")
            response_tokens = []
            token_count = 0
            max_response_tokens = MAX_RESPONSE_TOKENS
            
            # Response generation (streaming)
            gen_start = time.time()
            for token in llm.stream(prompt):
                print(token, end="", flush=True)
                response_tokens.append(token)
                token_count += len(token.split())
                
                # Prevent excessively long responses
                if token_count > max_response_tokens:
                    print("\n[dim]...(response truncated)[/dim]")
                    break
            
            gen_latency = (time.time() - gen_start) * 1000
            print(f"[dim]Generation latency: {gen_latency:.0f} ms[/dim]")
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
        epilog="For more information, visit: https://github.com/agvarun007/FinBot.git"
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
