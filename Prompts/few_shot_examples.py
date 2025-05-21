"""
Few-shot examples for the PowerBI LLM to help it understand how to formulate complex queries.
"""

from Prompts.query_examples import DAX_EXAMPLES

def get_few_shot_examples():
    """Format DAX examples for the LLM."""

    examples_text = "\n\nEXAMPLE QUERIES FOR REFERENCE:\n"
    examples_text += "Below are examples of DAX queries for different types of business questions.\n"
    examples_text += "Use these as reference patterns when constructing new queries or adapting existing ones.\n\n"
    examples_text += "IMPORTANT: When writing DAX queries, never include language identifiers or extra backticks.\n\n"

    for i, example in enumerate(DAX_EXAMPLES):
        # Create a formatted block for each example
        examples_text += f"--- Example {i+1} ---\n"
        examples_text += f"Question: \"{example['question']}\"\n"
        examples_text += "DAX Query:\n"  # Removed the triple backticks to avoid confusion
        examples_text += example['query'] + "\n\n"

    return examples_text