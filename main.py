"""
Main entry point for the Enhanced PowerBI Agent.
"""

import warnings
import os

# Suppress all warnings at the beginning of the script
warnings.filterwarnings("ignore")

# Suppress urllib3 warnings
os.environ["PYTHONWARNINGS"] = "ignore:Unverified HTTPS request"

# Import libraries after suppressing warnings
from langchain.agents import AgentExecutor, ZeroShotAgent, Tool
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

from custom_llm import CustomGPT
from config import AGENT_PREFIX, KNOWN_TABLES
from Prompts.few_shot_examples import get_few_shot_examples

from Tools.info_powerbi_tool import InfoPowerBITool
from Tools.list_powerbi_tool import ListPowerBITool
from Tools.schema_powerbi_tool import SchemaPowerBITool
from Tools.query_powerbi_tool import QueryPowerBITool

def create_powerbi_agent():
    """Create and configure the enhanced PowerBI agent."""

    # Initialize the custom LLM
    llm = CustomGPT()

    # Initialize tools
    tools = [
        InfoPowerBITool(),
        ListPowerBITool(),
        SchemaPowerBITool(),
        QueryPowerBITool()
    ]

    # Convert tools to a format ZeroShotAgent can use
    tool_names = [tool.name for tool in tools]

    # Create a separate string for few-shot examples to avoid template issues
    few_shot_examples = get_few_shot_examples()

    # Define the prefix (system message) for the agent
    prefix = f"""
    {AGENT_PREFIX}

    You are an advanced PowerBI data analyst specializing in complex queries and data analysis. Your expertise lies in understanding 
    star schema architectures with fact and dimension tables, and constructing appropriate joins to answer user questions.

    IMPORTANT CONTEXT ABOUT THE DATA MODEL:
    - The dataset follows a star schema with fact tables that contain measures and dimension tables that contain attributes.
    - To answer most user questions, you'll need to join tables using the appropriate keys.
    - Common business terms may not match database column names exactly. For example, "product" might be stored as "PRD_ID".
    - Always check table schemas and relationships before constructing queries.
    - Financial data is primarily stored in fact tables with monetary values in columns like "MONTANT".

    REASONING APPROACH - MATCH YOUR EXPLORATION DEPTH TO QUESTION COMPLEXITY:
    1. For simple questions (like "list tables", "show schema", "give me counts"), use minimal tool calls and return results immediately.
    2. For complex analytical questions, follow this more detailed approach:
       a. UNDERSTAND: First, understand what information the user is looking for and what business entities are involved.
       b. EXPLORE: Use info_powerbi_tool and schema_powerbi_tool to understand the relevant tables and their relationships.
       c. PLAN: Determine which tables need to be joined and which columns to use in the query.
       d. EXECUTE: Construct and run the appropriate query with necessary filters and calculations.
       e. EXPLAIN: Explain the results in business terms the user will understand.

    For complex queries involving multiple tables:
    - First identify the fact tables that contain the measures you need
    - Then identify the dimension tables needed for filtering or grouping
    - Use the relationship information to correctly join these tables
    - Apply appropriate filters and aggregations in your query

    DAX QUERY EXAMPLES:
    - To get distinct values from a column: EVALUATE VALUES(TableName[ColumnName])
    - To filter data: EVALUATE FILTER(TableName, TableName[Column] = "Value")
    - To get the top N records: EVALUATE TOPN(10, TableName)
    - To calculate a sum: EVALUATE SUMMARIZECOLUMNS(TableName[GroupColumn], "Total", SUM(TableName[AmountColumn]))
    
    When using query_powerbi_tool, always use DAX syntax, not SQL.
    """

    # Combine the prefix, examples, and tools list - ensuring examples are properly inserted
    full_prefix = prefix + "\n" + few_shot_examples + "\n\nYou have access to the following tools:"

    # Define the agent format instructions with strict guidance on format
    format_instructions = """Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    IMPORTANT EFFICIENCY RULES:
    - If you have all information needed to answer the question after initial tool call(s), proceed immediately to Final Answer.
    - Only perform additional exploratory tool calls if absolutely necessary to answer the question.
    - Do not run schema_powerbi_tool on tables unless you need their schema to answer the current question.
    - Match your exploration depth to question complexity - simple questions require minimal tool usage.

    IMPORTANT FORMAT RULES:
    - After each "Thought:", you MUST ALWAYS follow with either:
      - "Action:" (to use a tool), OR
      - "Final Answer:" (to provide the final response)
    - NEVER write a Thought without following it with either an Action or Final Answer.
    - ALWAYS use "Final Answer:" when you have the information needed to answer the user's question.

    NEVER invent or answer questions the user didn't ask.
    ALWAYS stay focused on the exact question the user is asking.
    """

    # Create the prompt template
    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=full_prefix,
        format_instructions=format_instructions,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )

    # Create the memory
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create the LLM chain
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Create the agent
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

    # Create the agent executor with increased max_iterations for complex queries
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        max_iterations=8,  # Increased to allow for more complex reasoning chains
        handle_parsing_errors=True,
        early_stopping_method="force",  # Force stopping on parsing errors to avoid infinite loops
    )

    return agent_executor

def main():
    # Create the agent
    agent = create_powerbi_agent()

    print("🤖 Your personal PowerBI Agent is initialized and Ready to Go ! ")
    print("📋 Ask questions about your PowerBI dataset or request analysis.")
    print("👋🏽 Type 'exit' to quit.\n")

    while True:
        user_input = input("• Question: ")

        if user_input.lower() == "exit":
            print("Goodbye! 👋🏽")
            break

        try:
            response = agent.invoke({"input": user_input})
            print(f"\nAnswer: {response['output']}\n")
        except Exception as e:
            print(f"Error: {str(e)}")
            print("\nAnswer: I encountered an issue processing your request. Please try rephrasing your question or ask something else.\n")

if __name__ == "__main__":
    main()

# Example usage:
# python3 main.py