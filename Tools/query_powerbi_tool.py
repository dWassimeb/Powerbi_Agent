"""
Tool for querying PowerBI tables.
"""

from langchain.tools import BaseTool
from typing import Dict, Any, Type
from utils.powerbi_helper import PowerBIHelper
from config import DATASET_ID
from pydantic import BaseModel, Field

# Define input schema for the tool
class QueryPowerBIToolInput(BaseModel):
    query: str = Field(..., description="The DAX query to execute against PowerBI tables.")

class QueryPowerBITool(BaseTool):
    name: str = "query_powerbi_tool"
    description: str = "Run a DAX query against PowerBI tables. Use DAX syntax, not SQL. For example, use 'EVALUATE VALUES(GL[PRODUIT])' instead of 'SELECT DISTINCT PRODUIT FROM GL'."
    args_schema: Type[BaseModel] = QueryPowerBIToolInput

    def _run(self, query: str) -> str:
        """Execute a DAX query against PowerBI dataset."""
        try:
            if not query:
                return "Error: Query is required."

            pbi_helper = PowerBIHelper(DATASET_ID)
            result = pbi_helper.execute_query(query)

            # If result is pandas DataFrame, convert to string
            if hasattr(result, "to_string"):
                return f"Query results:\n\n{result.to_string()}"
            else:
                return f"Query results:\n\n{result}"
        except Exception as e:
            return f"Error executing query: {str(e)}"

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