"""
Tool for querying PowerBI tables.
"""

from langchain.tools import BaseTool
from typing import Dict, Any, Type
from utils.powerbi_helper import PowerBIHelper
from config import DATASET_ID
from pydantic import BaseModel, Field
import re

# Define input schema for the tool
class QueryPowerBIToolInput(BaseModel):
    query: str = Field(..., description="The DAX query to execute against PowerBI tables.")

class QueryPowerBITool(BaseTool):
    name: str = "query_powerbi_tool"
    description: str = "Run a DAX query against PowerBI tables. Use DAX syntax, not SQL. All DAX queries must start with EVALUATE and must return a table expression. For simple calculations, use ROW() to wrap CALCULATE()."
    args_schema: Type[BaseModel] = QueryPowerBIToolInput

    def _clean_query(self, query: str) -> str:
        """Clean and prepare query string for execution."""
        # Remove leading/trailing whitespace
        query = query.strip()

        # Remove triple backticks and language identifiers
        if query.startswith("```"):
            # Find the first newline to skip the language identifier
            first_newline = query.find('\n')
            if first_newline > 0:
                # Remove everything before the first content line
                query = query[first_newline:].strip()
                # Also remove the closing backticks if present
                if query.endswith("```"):
                    query = query[:-3].strip()
        elif query.endswith("```"):
            query = query[:-3].strip()

        # Check if the query is wrapped in quotes and remove them
        if (query.startswith('"') and query.endswith('"')) or \
                (query.startswith("'") and query.endswith("'")):
            query = query[1:-1].strip()

        # Remove any language identifiers that appear after EVALUATE
        query = re.sub(r'EVALUATE\s+[A-Za-z]+\s+', 'EVALUATE ', query, flags=re.IGNORECASE)

        # Remove any nested EVALUATE statements
        if re.search(r'EVALUATE\s+EVALUATE', query, flags=re.IGNORECASE):
            query = re.sub(r'EVALUATE\s+', '', query, count=1, flags=re.IGNORECASE)

        # Ensure the query starts with EVALUATE
        if not re.search(r'^EVALUATE', query, re.IGNORECASE):
            query = "EVALUATE\n" + query

        # Replace escaped double quotes with single double quotes
        query = query.replace('""', '"')

        # Remove any tabs or irregular whitespace that could cause issues
        query = query.replace('\t', '    ')

        return query

    def _diagnose_syntax_error(self, query: str, error_message: str) -> str:
        """Diagnose common syntax errors and suggest corrections."""
        diagnoses = []

        # Check for language identifier after EVALUATE
        if re.search(r"EVALUATE\s+[A-Za-z]+\s+EVALUATE", query, re.IGNORECASE):
            diagnoses.append("Multiple EVALUATE statements or language identifier in query")

        # Check for missing RETURN in VAR queries
        if "VAR" in query.upper() and "RETURN" not in query.upper():
            diagnoses.append("Missing RETURN statement in VAR-based query")

        # Check for potentially unbalanced parentheses
        open_parens = query.count("(")
        close_parens = query.count(")")
        if open_parens != close_parens:
            diagnoses.append(f"Unbalanced parentheses: {open_parens} opening vs {close_parens} closing")

        if diagnoses:
            return "Syntax issues detected:\n- " + "\n- ".join(diagnoses)

        return "No common syntax issues detected. See error message for details."

    def _is_potentially_invalid_table_expression(self, query: str) -> bool:
        """Check if this query might be an invalid table expression."""
        # Extract the part after EVALUATE
        if "EVALUATE" in query.upper():
            body = re.sub(r"EVALUATE\s+", "", query, flags=re.IGNORECASE, count=1).strip()

            # Patterns that suggest direct CALCULATE use without table context
            if body.upper().startswith("CALCULATE(") and not any(func in body.upper() for func in ["ROW(", "SUMMARIZE", "SUMMARIZECOLUMNS", "VALUES", "TOPN", "DISTINCT"]):
                return True

        return False

    def _fix_invalid_table_expression(self, query: str) -> str:
        """Fix a query that might be an invalid table expression."""
        # Extract the part after EVALUATE
        body = re.sub(r"EVALUATE\s+", "", query, flags=re.IGNORECASE, count=1).strip()

        # If it starts with CALCULATE, wrap it in ROW
        if body.upper().startswith("CALCULATE("):
            # Extract any comments before CALCULATE
            comments = ""
            calculate_start = body.upper().find("CALCULATE(")
            if calculate_start > 0:
                comments = body[:calculate_start]
                body = body[calculate_start:]

            fixed_query = f"EVALUATE\n{comments}ROW(\"Result\", {body})"
            return fixed_query

        return query

    def _format_time_series(self, results):
        """Format time series data nicely."""
        formatted_output = "Time Series Results:\n\n"

        # Identify date/time column and measure columns
        time_cols = []
        for col in results[0].keys():
            col_lower = col.lower()
            if any(term in col_lower for term in ["mois", "date", "month", "year", "année", "trimestre", "semestre", "jour", "day"]):
                time_cols.append(col)

        time_col = time_cols[0] if time_cols else list(results[0].keys())[0]

        # Format as bullet points by time period
        for row in results:
            time_value = row[time_col]
            formatted_output += f"• {time_value}:\n"

            for col, val in row.items():
                if col != time_col:
                    if isinstance(val, (int, float)):
                        formatted_value = f"{val:,.2f}"
                    else:
                        formatted_value = str(val)
                    formatted_output += f"  - {col}: {formatted_value}\n"
            formatted_output += "\n"

        return formatted_output

    def _format_table(self, results):
        """Format general table data."""
        formatted_output = "Query Results:\n\n"

        # Get all column names
        columns = list(results[0].keys())

        # Format as bullet points
        for i, row in enumerate(results):
            formatted_output += f"• Row {i+1}:\n"
            for col in columns:
                value = row.get(col, "N/A")
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:,.2f}"
                else:
                    formatted_value = str(value)
                formatted_output += f"  - {col}: {formatted_value}\n"
            formatted_output += "\n"

        return formatted_output

    def _format_results(self, result):
        """Format query results in a more readable way."""
        if isinstance(result, dict):
            if "error" in result:
                return f"Error executing query: {result['error']}"

            if "results" in result and len(result["results"]) > 0:
                results = result["results"]

                # Case 1: Single value result (scalar)
                if len(results) == 1 and len(results[0]) == 1:
                    key = list(results[0].keys())[0]
                    value = results[0][key]

                    if isinstance(value, (int, float)):
                        return f"{key}: {value:,.2f}"
                    else:
                        return f"{key}: {value}"

                # Case 2: Time series data
                if len(results) > 1 and any(
                    any(term in col.lower() for term in ["mois", "date", "month", "year", "année"])
                    for col in results[0].keys()
                ):
                    return self._format_time_series(results)

                # Case 3: General table data
                return self._format_table(results)

            return "Query returned no results."

        return f"Query results:\n\n{str(result)}"

    def _run(self, query: str) -> str:
        """Execute a DAX query against PowerBI dataset with automatic correction."""
        try:
            if not query:
                return "Error: Query is required."

            # Step 1: Clean the query
            original_query = query
            cleaned_query = self._clean_query(query)

            # Print both for debugging
            print("\nOriginal query:")
            print(original_query)
            print("\nCleaned query:")
            print(cleaned_query)

            # Step 2: Check if this might be an invalid table expression
            if self._is_potentially_invalid_table_expression(cleaned_query):
                # Fix it proactively
                fixed_query = self._fix_invalid_table_expression(cleaned_query)
                if fixed_query != cleaned_query:
                    print("\nCorrected invalid table expression:")
                    print(fixed_query)
                    cleaned_query = fixed_query

            try:
                # Step 3: Execute the query
                print("\nExecuting query:")
                print(cleaned_query)
                pbi_helper = PowerBIHelper(DATASET_ID)
                result = pbi_helper.execute_query(cleaned_query)
                return self._format_results(result)

            except Exception as first_error:
                # Step 4: If it fails with specific error, try to fix it
                error_message = str(first_error)
                print(f"\nEncountered error: {error_message}")

                # Diagnose the syntax error
                syntax_diagnosis = self._diagnose_syntax_error(cleaned_query, error_message)
                print(f"Diagnosis: {syntax_diagnosis}")

                if "not a valid table expression" in error_message:
                    print("\nAttempting to fix invalid table expression...")

                    # Try to fix the query
                    fixed_query = self._fix_invalid_table_expression(cleaned_query)
                    if fixed_query != cleaned_query:
                        print("\nFixed query:")
                        print(fixed_query)

                        # Try again with the fixed query
                        result = pbi_helper.execute_query(fixed_query)
                        return self._format_results(result)

                # If we can't fix it or it fails for another reason, return the original error with diagnosis
                return f"Error executing query: {error_message}\n\nDiagnosis: {syntax_diagnosis}\n\nAttempted query:\n{cleaned_query}"

        except Exception as e:
            return f"Error executing query: {str(e)}\n\nAttempted query:\n{query}"

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert the tool to an OpenAI function."""
        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The DAX query to execute against PowerBI tables. Use DAX syntax, not SQL."
                    }
                },
                "required": ["query"]
            }
        }
        return schema