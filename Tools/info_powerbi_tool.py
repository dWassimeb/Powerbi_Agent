"""
Enhanced tool for retrieving detailed PowerBI metadata information.
"""

from langchain.tools import BaseTool
from typing import Dict, Any, Type, Optional
from pydantic import BaseModel, Field
from config import DATASET_ID, KNOWN_TABLES
from utils.powerbi_helper import PowerBIHelper

# Import the metadata
from data.metadata import DATASET_METADATA, RELATIONSHIPS

class InfoPowerBIToolInput(BaseModel):
    table_name: Optional[str] = Field(None, description="Optional: The specific table name to get detailed information about. If not provided, returns information about all tables and their relationships.")

class InfoPowerBITool(BaseTool):
    name: str = "info_powerbi_tool"
    description: str = "Get detailed information about PowerBI tables including descriptions, relationships, and column details. Useful for understanding data structure before querying."
    args_schema: Type[BaseModel] = InfoPowerBIToolInput

    def _run(self, table_name: Optional[str] = None) -> str:
        """
        Return detailed information about PowerBI dataset tables, their relationships, and columns.
        If table_name is specified, returns detailed information about that specific table.
        """
        try:
            # If a specific table is requested
            if table_name:
                if table_name not in DATASET_METADATA:
                    available_tables = ", ".join(DATASET_METADATA.keys())
                    return f"Table '{table_name}' not found in metadata. Available tables are: {available_tables}"

                # Get table metadata
                table_info = DATASET_METADATA.get(table_name, {})
                description = table_info.get("description", "No description available")
                columns = table_info.get("columns", {})
                table_type = table_info.get("type", "Unknown type")

                # Get relationships for this table
                table_relationships = []
                for rel in RELATIONSHIPS:
                    if rel["from_table"] == table_name or rel["to_table"] == table_name:
                        table_relationships.append(rel)

                # Format the output
                result = f"## Table: {table_name}\n\n"
                result += f"Description: {description}\n\n"
                result += f"Table Type: {table_type.capitalize()} table\n\n"

                # Add column information
                result += "### Columns:\n"
                for col_name, col_desc in columns.items():
                    result += f"- {col_name}: {col_desc if col_desc != '...' else 'No description'}\n"

                # Add relationship information
                if table_relationships:
                    result += "\n### Relationships:\n"
                    for rel in table_relationships:
                        rel_desc = rel.get("description", "")
                        if rel["from_table"] == table_name:
                            result += f"- This table ({table_name}.{rel['from_key']}) → {rel['to_table']}.{rel['to_key']} {rel_desc}\n"
                        else:
                            result += f"- {rel['from_table']}.{rel['from_key']} → This table ({table_name}.{rel['to_key']}) {rel_desc}\n"

                return result

            # If no specific table is requested, return overview of data model
            else:
                result = "# PowerBI Dataset Overview\n\n"

                # Group tables by type
                fact_tables = [t for t, info in DATASET_METADATA.items() if info.get("type") == "fact"]
                dimension_tables = [t for t, info in DATASET_METADATA.items() if info.get("type") == "dimension"]

                result += "## Table Types:\n"
                result += "- Fact tables: " + ", ".join(fact_tables) + "\n"
                result += "- Dimension tables: " + ", ".join(dimension_tables) + "\n\n"

                result += "## Key Tables:\n"
                for table_name, table_info in DATASET_METADATA.items():
                    result += f"- {table_name} ({table_info.get('type', 'unknown').capitalize()}): {table_info.get('description', 'No description')}\n"

                result += "\n## Key Relationships:\n"
                for rel in RELATIONSHIPS:
                    rel_desc = rel.get("description", "")
                    result += f"- {rel['from_table']}.{rel['from_key']} → {rel['to_table']}.{rel['to_key']} {rel_desc}\n"

                result += "\n## Data Model Architecture:\n"
                result += "This dataset uses a star schema architecture with fact tables containing measures, connected to various dimension tables through foreign keys. To perform complex queries, you'll need to join tables using these relationships.\n\n"

                result += "To get detailed information about a specific table, use the info_powerbi_tool with a table name parameter."

                return result

        except Exception as e:
            return f"Error retrieving table information: {str(e)}"

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
                        "description": "Optional: The specific table name to get detailed information about. If not provided, returns information about all tables and their relationships."
                    }
                },
                "required": []
            }
        }
        return schema