# Tools/match_query_example_tool.py
"""
Tool for matching user queries to example DAX queries.
"""

from langchain.tools import BaseTool
from typing import Dict, Any, Type
from pydantic import BaseModel, Field
from Prompts.query_examples import DAX_EXAMPLES

class MatchQueryExampleToolInput(BaseModel):
    question: str = Field(..., description="The user's question to match against example queries.")

class MatchQueryExampleTool(BaseTool):
    name: str = "match_query_example_tool"
    description: str = "Find if a user's question matches or is similar to any predefined DAX query examples. Returns the matching query if found."
    args_schema: Type[BaseModel] = MatchQueryExampleToolInput

    def _run(self, question: str) -> str:
        """Find matching query examples for a given question."""
        if not question:
            return "Error: Question is required."

        # Normalize input for comparison
        normalized_input = question.lower().strip()

        # Check for exact matches
        for example in DAX_EXAMPLES:
            if normalized_input == example['question'].lower().strip():
                return f"EXACT_MATCH: Found exact match with template question: '{example['question']}'\nUse this query:\n\n{example['query']}"

        # Check for similar matches
        best_match = None
        highest_similarity = 0.5

        for example in DAX_EXAMPLES:
            # Compare based on key terms
            example_terms = set(example['question'].lower().replace('?', '').split())
            user_terms = set(normalized_input.replace('?', '').split())

            # Calculate similarity
            if len(example_terms) > 0:
                common_terms = example_terms.intersection(user_terms)
                similarity = len(common_terms) / len(example_terms)

                # Find potential parameters that need adaptation
                params_to_adapt = []

                # Look for year differences
                if "2023" in example['question'] and "2023" not in normalized_input:
                    for year in ["2020", "2021", "2022", "2024", "2025"]:
                        if year in normalized_input:
                            params_to_adapt.append(f"Year: from 2023 to {year}")

                if "2024" in example['question'] and "2024" not in normalized_input:
                    for year in ["2020", "2021", "2022", "2023", "2025"]:
                        if year in normalized_input:
                            params_to_adapt.append(f"Year: from 2024 to {year}")

                # Look for TOP N differences
                import re
                example_top = re.search(r"TOP\s+(\d+)", example['question'], re.IGNORECASE)
                user_top = re.search(r"TOP\s+(\d+)", normalized_input, re.IGNORECASE)

                if example_top and user_top and example_top.group(1) != user_top.group(1):
                    params_to_adapt.append(f"TOP N: from {example_top.group(1)} to {user_top.group(1)}")

                # Look for product differences
                if "P231" in example['question'] and "P231" not in normalized_input:
                    # Find potential product codes in user query
                    product_match = re.search(r"[Pp]\d{3}", normalized_input)
                    if product_match:
                        params_to_adapt.append(f"Product Code: from P231 to {product_match.group()}")

                # Look for business unit differences
                if "Digital Solutions" in example['question'] and "Digital Solutions" not in normalized_input:
                    # Try to extract BU name from query
                    bu_words = ["BU", "sous bu", "business unit"]
                    for bu_word in bu_words:
                        if bu_word in normalized_input:
                            bu_pos = normalized_input.find(bu_word) + len(bu_word)
                            possible_bu = normalized_input[bu_pos:bu_pos+30].strip()
                            if possible_bu:
                                params_to_adapt.append(f"Business Unit: from Digital Solutions to a different BU")

                # If better than previous matches
                if similarity > highest_similarity:
                    highest_similarity = similarity

                    # Prepare result information
                    match_info = {
                        "similarity": similarity,
                        "template_question": example['question'],
                        "query": example['query'],
                        "params_to_adapt": params_to_adapt
                    }

                    # Format the result
                    result = f"SIMILAR_MATCH: Found similar match with template question: '{example['question']}'\n"
                    result += f"Similarity score: {similarity:.0%}\n"

                    if params_to_adapt:
                        result += "Parameters that may need adaptation:\n"
                        for param in params_to_adapt:
                            result += f"- {param}\n"

                    result += f"\nTemplate query:\n\n{example['query']}"
                    best_match = result

        if best_match:
            return best_match

        return "NO_MATCH: No matching query examples found. Create a new DAX query based on the tables and relationships."

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert the tool to an OpenAI function."""
        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The user's question to match against example queries."
                    }
                },
                "required": ["question"]
            }
        }
        return schema