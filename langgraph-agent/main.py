from agent import build_graph

def run(question: str):
    print(f"\n{'='*50}")
    print(f"Question: {question}")
    print('='*50)

    graph = build_graph()
    result = graph.invoke({
        "question": question,
        "messages": [{"role": "user", "content": question}],
        "tool_calls": [],
        "final_answer": "",
    })

    # extract final answer from last assistant message
    final = ""
    for m in reversed(result["messages"]):
        if m["role"] == "assistant" and "ANSWER:" in m["content"]:
            final = m["content"].split("ANSWER:")[1].strip()
            break

    print(f"\n{'='*50}")
    print(f"Final Answer: {final}")
    print(f"Tools used: {[t['tool'] for t in result['tool_calls']]}")
    print('='*50)

if __name__ == "__main__":
    run("What is the weight capacity of the Hexapod X1, and what is that in pounds? (1 kg = 2.205 lbs)")