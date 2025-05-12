"""
Tool for getting schema of PowerBI tables.
"""

from langchain.tools import BaseTool
from utils.powerbi_helper import PowerBIHelper
from config import DATASET_ID, KNOWN_TABLES
from typing import Dict, Any, Type
from pydantic import BaseModel, Field

# Define input schema for the tool
class SchemaPowerBIToolInput(BaseModel):
    table_name: str = Field(..., description="The name of the table to get the schema for.")

class SchemaPowerBITool(BaseTool):
    name: str = "schema_powerbi_tool"
    description: str = "Get the schema of a specific PowerBI table."
    args_schema: Type[BaseModel] = SchemaPowerBIToolInput

    # Define the pbi_helper attribute at the class level
    pbi_helper: Any = None

    def __init__(self):
        """Initialize the schema tool."""
        super().__init__()
        # Create a PowerBI helper instance
        self.pbi_helper = PowerBIHelper(DATASET_ID)

    def _run(self, table_name: str) -> str:
        """Get schema for a specific table."""
        try:
            if not table_name:
                return "Error: Table name is required."

            # Get the schema using the PowerBIHelper
            schema = self.pbi_helper.get_table_schema(table_name)

            # Format the response
            if isinstance(schema, dict) and "error" in schema:
                return f"Could not retrieve schema for table '{table_name}'. {schema['error']}"

            if "columns" not in schema or not schema["columns"]:
                return f"No columns found for table '{table_name}'."

            response = f"Schema for table '{table_name}':\n\n"
            response += "Columns:\n"

            for column in schema.get("columns", []):
                response += f"- {column['name']} ({column.get('dataType', 'unknown type')})\n"

            return response
        except Exception as e:
            return f"Error getting schema for table '{table_name}': {str(e)}"

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert the tool to an OpenAI function."""
        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table to get the schema for."
                    }
                },
                "required": ["table_name"]
            }
        }
        return schema