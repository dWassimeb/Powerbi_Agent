"""
Configuration settings for the PowerBI Agent.
"""

# PowerBI Dataset ID and workspace id
DATASET_ID = "b715c872-443b-42a7-b5d0-4cc9f92bd88b"
workspace_id = "7ab6eef2-3720-409f-9dee-d5cd868c559e"

# Known tables in the dataset
KNOWN_TABLES = [
    "INDICATEURS", "DIM_CLIENT", "DIM_DATE", "DIM_SOCIETE",
    "GL", "MAPPING_ACTIVITE", "MAPPING_CDR", "MAPPING_COMPTE",
    "MAPPING_PRODUIT", "MAPPING_PROJET_CDR", "MAPPING_PROJECT_MULTIBU"
]

# Agent configuration
AGENT_PREFIX = """You are a PowerBI assistant that helps users query and analyze data. 
You have access to a PowerBI dataset with tables containing financial and organizational data.
When answering questions, first gather information about the tables and their relationships, 
then formulate and execute appropriate queries to get the required information."""

AGENT_FORMAT_INSTRUCTIONS = """ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

# Tool descriptions
TOOL_DESCRIPTIONS = {
    "info": "Get detailed information about PowerBI tables including descriptions and relationships",
    "list": "List all available PowerBI tables in the dataset",
    "schema": "Get the schema of a specific PowerBI table",
    "query": "Run a SQL query against PowerBI tables"
}