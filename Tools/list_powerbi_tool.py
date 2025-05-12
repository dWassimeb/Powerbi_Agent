"""
Tool for listing PowerBI tables.
"""

from langchain.tools import BaseTool
from utils.powerbi_helper import PowerBIHelper
from config import DATASET_ID, KNOWN_TABLES
from typing import Dict, Any, Type
from pydantic import BaseModel

class ListPowerBIToolInput(BaseModel):
    pass

class ListPowerBITool(BaseTool):
    name: str = "list_powerbi_tool"
    description: str = "List all available PowerBI tables in the dataset."
    args_schema: Type[BaseModel] = ListPowerBIToolInput

    def _run(self) -> str:
        """List all tables in the PowerBI dataset."""
        try:
            # Initialize with known tables from config
            pbi_helper = PowerBIHelper(DATASET_ID)
            tables = pbi_helper.list_tables()

            # Format the response
            if not tables:
                return "No tables found in the PowerBI dataset."

            response = "Available tables in the PowerBI dataset:\n\n"
            for table in tables:
                response += f"- {table['name']}\n"

            return response
        except Exception as e:
            # Fallback to returning hardcoded known tables if there's an error
            response = "Available tables in the PowerBI dataset:\n\n"
            for table_name in KNOWN_TABLES:
                response += f"- {table_name}\n"
            return response

    def to_openai_function(self) -> Dict[str, Any]:
        """Convert the tool to an OpenAI function."""
        schema = {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {},  # No parameters needed
                "required": []
            }
        }
        return schema