import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text:str) -> int:
    # print(encoding.encode(text))
    return len(encoding.encode(text))

text = "I am building an AI-powered document chatbot."

print("Text:", text)
print("Token count:", count_tokens(text))  

CONTEXT_LIMIT = 8192
MAX_OUTPUT_TOKENS = 1000
SAFETY_MARGIN = 500

INPUT_BUDGET  = (
    CONTEXT_LIMIT
    - MAX_OUTPUT_TOKENS
    - SAFETY_MARGIN
)
print("Available input budget:", INPUT_BUDGET)

def select_chunks(chunks, token_budget):
    selected = []
    used_tokens = 0

    for chunk in chunks:
        chunk_tokens = count_tokens(chunk)
        # print(f'chunk tokens for {chunk} -> {chunk_tokens} and current {used_tokens}')
        if used_tokens + chunk_tokens <= token_budget:
            selected.append(chunk)
            used_tokens += chunk_tokens
            # print('chunk selected')
            # print("------------------------")

    return selected, used_tokens

chunks = [  
    "The company's revenue increased by 20%.",
    "Operating expenses increased by 12%.",
    "Net profit reached 50 million.",
    "The company expanded into three new markets."
]

selected, total_tokens = select_chunks(
    chunks,
    token_budget=100
)

print("Selected chunks:", len(selected))
print("Total tokens:", total_tokens)

def build_context(
    question,
    retrieved_chunks,
    system_prompt,
    input_budget
):
    selected_chunks = []
    used_tokens = count_tokens(system_prompt)
    used_tokens += count_tokens(question)

    for chunk in retrieved_chunks:

        chunk_tokens = count_tokens(chunk)

        if used_tokens + chunk_tokens <= input_budget:
            selected_chunks.append(chunk)
            used_tokens += chunk_tokens

    context = "\n\n".join(selected_chunks)

    return {
        "context": context,
        "selected_chunks": len(selected_chunks),
        "total_input_tokens": used_tokens
    }

system_prompt = """
You are a document question-answering assistant.
Answer only using the supplied context.
If the answer is unavailable, say so.
"""

question = "What was the company's revenue?"

retrieved_chunks = [
    "Revenue increased by 20% in FY2025.",
    "Net profit was 50 million.",
    "Operating expenses increased by 12%."
]

result = build_context(
    question=question,
    retrieved_chunks=retrieved_chunks,
    system_prompt=system_prompt,
    input_budget=4000
)

print(result)