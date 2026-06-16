import json
import os
from datetime import datetime
from ingest import load_documents
from agent import process_document

def generate_report(results: list[dict], output_path: str):
    lines = []
    lines.append("# Document Processing Report")
    lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Documents processed: {len(results)}\n")
    lines.append("---\n")

    # summary table
    lines.append("## Overview\n")
    lines.append("| Document | Title | Category |")
    lines.append("|---|---|---|")
    for r in results:
        title = r.get("title", "unknown")
        category = r.get("category", "unknown")
        filename = r.get("filename", "")
        lines.append(f"| {filename} | {title} | {category} |")

    lines.append("\n---\n")

    # per document details
    lines.append("## Detailed Analysis\n")
    for r in results:
        if "error" in r:
            lines.append(f"### {r['filename']}\n")
            lines.append(f"⚠️ Processing failed: {r.get('error')}\n")
            continue

        lines.append(f"### {r.get('title', r['filename'])}\n")
        lines.append(f"**File:** {r['filename']}  ")
        lines.append(f"**Category:** {r.get('category', 'unknown')}  ")
        lines.append(f"**Size:** {r['num_chars']:,} chars, {r['num_chunks']} chunks\n")

        lines.append(f"**Summary:**  \n{r.get('summary', 'N/A')}\n")

        lines.append(f"**Novelty:**  \n{r.get('novelty', 'N/A')}\n")

        methods = r.get("methods", [])
        if methods:
            lines.append(f"**Methods:** {', '.join(methods)}\n")

        topics = r.get("topics", [])
        if topics:
            lines.append(f"**Topics:** {', '.join(topics)}\n")

        lines.append("---\n")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Report written to {output_path}")


def main():
    print("=== Agentic Document Processor ===\n")

    # load documents
    docs = load_documents("docs/")
    print(f"\nLoaded {len(docs)} documents\n")

    # process each document
    results = []
    for i, doc in enumerate(docs):
        print(f"[{i+1}/{len(docs)}] Processing {doc['filename']}...")
        result = process_document(doc)
        results.append(result)

    # save raw JSON
    os.makedirs("output", exist_ok=True)
    json_path = "output/results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {json_path}")

    # generate report
    report_path = "output/report.md"
    generate_report(results, report_path)

    print("\n=== Done ===")
    print(f"  Results: {json_path}")
    print(f"  Report:  {report_path}")


if __name__ == "__main__":
    main()