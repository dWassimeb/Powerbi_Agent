"""
Few-shot examples for the PowerBI LLM to help it understand how to formulate complex queries.
"""

from Prompts.query_examples import DAX_EXAMPLES

def get_few_shot_examples():
    """Format DAX and SQL examples for inclusion in the agent's prompt."""

    examples_text = "Example DAX queries:\n"

    for example in DAX_EXAMPLES :  # Using just 2 examples to keep prompt size reasonable  DAX_EXAMPLES[:2]
        examples_text += f"\nQuestion: {example['question']}\n"
        # Replace curly braces with double curly braces to escape them
        escaped_query = example['query'].replace("{", "{{").replace("}", "}}")
        examples_text += f"Query:\n{escaped_query.strip()}\n"

    return examples_text