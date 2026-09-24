"""
Interactive hybrid-retrieval search over the watched-film corpus -- no LLM
involved, just cinemagent.retrieval's vector + BM25 -> RRF -> rerank.

Prints the final reranked list with each film's raw cross-encoder logit and
its confidence (sigmoid of the logit, as retrieve() returns). --stages also
prints the vector-only, BM25-only and fused (RRF) lists first, to inspect
the pipeline stage by stage.

Usage:
    python3 cli/search.py [--stages]
    (type a query, press enter, repeat; blank line or "quit" to exit)
"""

import argparse

from cinemagent.config import RERANK_TOP_N
from cinemagent.retrieval import load_retrieval_index, retrieve_stages, sigmoid


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", action="store_true",
                        help="Also print the vector-only, BM25-only and fused (RRF) lists")
    args = parser.parse_args()

    index = load_retrieval_index(verbose=True)

    def describe(film_id):
        film = index.film_lookup.get(film_id)
        return f"{film['title']} ({film.get('year', '?')})" if film else film_id

    print("Ready. Type a query, blank line or 'quit' to exit.\n")

    while True:
        query = input("query> ").strip()
        if not query or query.lower() in ("quit", "exit"):
            break

        stages = retrieve_stages(index, query)

        if args.stages:
            print("\n-- vector-only --")
            for rank, fid in enumerate(stages.vector_ids, 1):
                print(f"  {rank}. {describe(fid)}")

            print("-- bm25-only --")
            for rank, fid in enumerate(stages.bm25_ids, 1):
                print(f"  {rank}. {describe(fid)}")

            print("-- fused (RRF) --")
            for rank, (fid, score) in enumerate(stages.fused, 1):
                print(f"  {rank}. {describe(fid)}  rrf_score={score:.4f}")

        print(f"\n-- reranked (top {RERANK_TOP_N}, cross-encoder) --")
        for rank, (fid, score) in enumerate(stages.reranked, 1):
            print(f"  {rank}. {describe(fid)}  logit={score:+.4f}  confidence={sigmoid(float(score)):.4f}")
        print()


if __name__ == "__main__":
    main()
