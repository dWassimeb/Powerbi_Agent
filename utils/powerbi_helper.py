"""
Helper class for interacting with PowerBI APIs and data.
"""

import requests
import urllib.parse
from azure.identity import InteractiveBrowserCredential
from config import DATASET_ID, KNOWN_TABLES

class PowerBIHelper:
    def __init__(self, dataset_id=None, table_names=None):
        """Initialize the PowerBI Helper with dataset ID and credentials."""
        self.dataset_id = dataset_id or DATASET_ID
        self.credential = InteractiveBrowserCredential()
        self.token = self.get_token()
        self.base_url = "https://api.powerbi.com/v1.0/myorg"
        self.table_names = table_names or KNOWN_TABLES  # Use config tables if none provided

        # Remove explicit workspace_id to match LangChain implementation
        # Let the API determine the appropriate workspace

    def get_token(self):
        """Get authentication token for Power BI API."""
        token = self.credential.get_token("https://analysis.windows.net/powerbi/api/.default")
        return token.token

    def get_headers(self):
        """Get headers for API requests including authentication token."""
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

    def refresh_token_if_needed(self, response):
        """Refresh token if response indicates authentication issues."""
        if response.status_code == 401:
            self.token = self.get_token()
            return True
        return False

    def get_table_schema(self, table_name):
        """Get schema information for a specific table using DAX query."""
        try:
            # Use DAX query to get a single row, then extract column names
            query = f"""
            EVALUATE
            TOPN(1, {table_name})
            """

            result = self.execute_query(query)

            if isinstance(result, dict) and "error" in result:
                return {"error": f"Failed to execute query: {result['error']}"}

            # Process the results
            columns = []
            if "results" in result and result["results"] and len(result["results"]) > 0:
                # Extract column names from the first row
                row = result["results"][0]
                for col_name, value in row.items():
                    # Infer data type from value
                    data_type = "string"  # Default
                    if isinstance(value, (int, float)):
                        data_type = "number"
                    elif isinstance(value, bool):
                        data_type = "boolean"
                    elif isinstance(value, dict) and value.get("__type__") == "datetime":
                        data_type = "datetime"

                    columns.append({
                        "name": col_name,
                        "dataType": data_type
                    })

            return {"columns": columns}
        except Exception as e:
            return {"error": f"Error getting schema: {str(e)}"}

    def list_tables(self):
        """List all tables in the PowerBI dataset."""
        try:
            # Match LangChain's approach - use the dataset directly without workspace
            endpoint = f"{self.base_url}/datasets/{self.dataset_id}/tables"
            response = requests.get(endpoint, headers=self.get_headers())

            # Handle token refresh if needed
            if self.refresh_token_if_needed(response):
                response = requests.get(endpoint, headers=self.get_headers())

            if response.status_code != 200:
                return {"error": f"Failed to execute API request: {response.status_code} - {response.text}"}

            # Process the response
            tables_data = response.json()
            tables = tables_data.get("value", [])

            table_list = [table.get("name", "") for table in tables]
            return table_list
        except Exception as e:
            return {"error": f"Error listing tables: {str(e)}"}

    def execute_query(self, query):
        """Execute a DAX query against the PowerBI dataset."""
        try:
            # Match LangChain's approach - use the dataset directly without workspace
            endpoint = f"{self.base_url}/datasets/{self.dataset_id}/executeQueries"

            body = {
                "queries": [
                    {
                        "query": query
                    }
                ],
                "serializerSettings": {
                    "includeNulls": True
                }
            }

            response = requests.post(endpoint, headers=self.get_headers(), json=body)

            # Handle token refresh if needed
            if self.refresh_token_if_needed(response):
                response = requests.post(endpoint, headers=self.get_headers(), json=body)

            if response.status_code != 200:
                return {"error": f"Failed to execute query: {response.status_code} - {response.text}"}

            # Process the response
            result = response.json()

            if "results" in result and len(result["results"]) > 0:
                # Return the results
                tables = result["results"][0].get("tables", [])
                if tables and len(tables) > 0:
                    return {"results": tables[0].get("rows", [])}
                else:
                    return {"results": []}
            else:
                return {"results": []}
        except Exception as e:
            return {"error": f"Error executing query: {str(e)}"}