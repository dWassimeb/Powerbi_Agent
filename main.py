"""
Main entry point for the Enhanced PowerBI Agent with improved measure handling.
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
from Tools.match_query_example_tool import MatchQueryExampleTool

# Special sections for enhanced system prompt
YOY_SYSTEM_PROMPT = """
SPECIAL GUIDANCE FOR YEAR-OVER-YEAR (YoY) QUERIES AND N-1 MEASURES:

When handling Year-over-Year (YoY) comparisons or using N-1 measures, follow these specific guidelines:

1. CORRECT USAGE OF [CA/N-1] MEASURE:
   - The [CA/N-1] measure is designed to show the previous year's revenue for the CURRENT period
   - When using [CA/N-1], filter DIM_DATE for the CURRENT year, not the previous year
   - CORRECT: CALCULATE([CA/N-1], DIM_DATE[Année] = 2024, DIM_DATE[TRIMESTRE] = 3)
   - INCORRECT: CALCULATE([CA/N-1], DIM_DATE[Année] = 2023, DIM_DATE[TRIMESTRE] = 3)

2. SIMPLIFIED APPROACH FOR YoY COMPARISONS:
   If encountering errors with complex YoY queries, use this simpler alternative structure:
   
   EVALUATE
   ROW(
       "Current Year", CALCULATE([CA], DIM_DATE[Année] = 2024, DIM_DATE[TRIMESTRE] = 3, GL[BU] = "HCS"),
       "Previous Year", CALCULATE([CA], DIM_DATE[Année] = 2023, DIM_DATE[TRIMESTRE] = 3, GL[BU] = "HCS"),
       "Difference", CALCULATE([CA], DIM_DATE[Année] = 2024, DIM_DATE[TRIMESTRE] = 3, GL[BU] = "HCS") - 
                     CALCULATE([CA], DIM_DATE[Année] = 2023, DIM_DATE[TRIMESTRE] = 3, GL[BU] = "HCS"),
       "Percent Change", DIVIDE(
           CALCULATE([CA], DIM_DATE[Année] = 2024, DIM_DATE[TRIMESTRE] = 3, GL[BU] = "HCS") - 
           CALCULATE([CA], DIM_DATE[Année] = 2023, DIM_DATE[TRIMESTRE] = 3, GL[BU] = "HCS"),
           CALCULATE([CA], DIM_DATE[Année] = 2023, DIM_DATE[TRIMESTRE] = 3, GL[BU] = "HCS"),
           0
       )
   )

3. HANDLING DIFFERENCE MEASURES:
   - The [DIF CA/CA_N-1] measure calculates the difference between current and previous year values
   - Always filter by the CURRENT year when using difference measures
   - CORRECT: CALCULATE([DIF CA/CA_N-1], DIM_DATE[Année] = 2024, DIM_DATE[TRIMESTRE] = 3)
   - INCORRECT: Trying to calculate differences across multiple years in a single CALCULATE function

4. FIXING COMMON YoY QUERY ERRORS:
   If you encounter errors with complex YoY queries using measures like [CA/N-1] or [DIF CA/CA_N-1]:
   
   a. First try the simplified single-row approach shown in point #2
   
   b. If that fails, try splitting into two separate queries:
      - Query 1: Get current year values only
      - Query 2: Get previous year values only
      - Then present both results to the user
   
   c. For percentage calculations, always use DIVIDE() with a fallback value (0):
      DIVIDE(current_value - previous_value, previous_value, 0)

5. AVOIDING MEASURE SYNTAX ERRORS:
   - Always use correct syntax for N-1 measures: [CA/N-1], NOT [CA]/N-1 or [[CA]/N-1]
   - Always use correct syntax for difference measures: [DIF CA/CA_N-1], NOT [DIF [CA]/CA_N-1]
   - Ensure you never have double brackets [[...]] in your measures

EXAMPLE YoY QUERY PATTERN THAT WORKS CONSISTENTLY:

```
EVALUATE
ROW(
    "Q3 2024 Revenue for HCS", CALCULATE(
        [CA],
        GL[BU] = "HCS",
        DIM_DATE[Année] = 2024,
        DIM_DATE[TRIMESTRE] = 3
    ),
    "Q3 2023 Revenue for HCS", CALCULATE(
        [CA],
        GL[BU] = "HCS",
        DIM_DATE[Année] = 2023,
        DIM_DATE[TRIMESTRE] = 3
    ),
    "Year-over-Year Difference", CALCULATE(
        [CA],
        GL[BU] = "HCS",
        DIM_DATE[Année] = 2024,
        DIM_DATE[TRIMESTRE] = 3
    ) - CALCULATE(
        [CA],
        GL[BU] = "HCS",
        DIM_DATE[Année] = 2023,
        DIM_DATE[TRIMESTRE] = 3
    )
)
```

This approach is more reliable than trying to use multiple measures within SUMMARIZECOLUMNS for YoY comparisons.
"""

TABLE_COLUMN_GUIDANCE = """
KEY TABLE-COLUMN MAPPINGS TO REMEMBER:

1. Always filter Business Units (BU) using GL table:
   - Correct: GL[BU] = "HCS"
   - Incorrect: DIM_SOCIETE[BU] = "HCS" (BU column doesn't exist in DIM_SOCIETE)

2. Always filter Sub-Business Units using GL table:
   - Correct: GL[Sous BU] = "Digital Solutions"
   - Incorrect: DIM_SOCIETE[Sous BU] = "Digital Solutions"

3. Always use DIM_DATE for time-based filtering:
   - For years: DIM_DATE[Année] = 2024
   - For quarters: DIM_DATE[TRIMESTRE] = 3 
   - For months: DIM_DATE[MOIS] = 9
   - Always filter by numerical values, not text

4. Always filter clients using DIM_CLIENT:
   - Correct: DIM_CLIENT[CLIENT_NOM] = "Acme Corp"
   - Link with GL: Through GL[Client] = DIM_CLIENT[CLIENT_ID]

5. Always filter products using MAPPING_PRODUIT:
   - Correct: MAPPING_PRODUIT[Produit] = "Product Name" or MAPPING_PRODUIT[Code Produit] = "P123"
   - Link with GL: Through GL[PRODUIT] = MAPPING_PRODUIT[Code Produit]

6. Avoid using CALCULATE with FILTER within SUMMARIZECOLUMNS:
   - Prefer: SUMMARIZECOLUMNS(table[column], "Measure", CALCULATE([Measure], conditions...))
   - Instead of: SUMMARIZECOLUMNS(FILTER(table, conditions...), "Measure", [Measure])

7. For date hierarchies, remember these columns exist:
   - DIM_DATE[Année]: Year number
   - DIM_DATE[TRIMESTRE]: Quarter number (1-4)
   - DIM_DATE[MOIS]: Month number (1-12)
   - DIM_DATE[MOIS_NOM]: Month name
   - DIM_DATE[SEMESTRE]: Semester (1-2)
"""

IMPROVED_PATTERNS = """
IMPROVED DAX QUERY PATTERNS FOR COMMON QUESTIONS:

1. Revenue for specific Business Unit in a period:
   EVALUATE
   ROW(
       "Revenue for BU in Period", CALCULATE(
           [CA],
           GL[BU] = "HCS",
           DIM_DATE[Année] = 2024,
           DIM_DATE[TRIMESTRE] = 3
       )
   )

2. Year-over-Year comparison:
   EVALUATE
   ROW(
       "Current Year", CALCULATE([CA], DIM_DATE[Année] = 2024, GL[BU] = "HCS"),
       "Previous Year", CALCULATE([CA], DIM_DATE[Année] = 2023, GL[BU] = "HCS"),
       "YoY Difference", CALCULATE([CA], DIM_DATE[Année] = 2024, GL[BU] = "HCS") - 
                         CALCULATE([CA], DIM_DATE[Année] = 2023, GL[BU] = "HCS")
   )

3. Monthly trend for current year:
   EVALUATE
   SUMMARIZECOLUMNS(
       DIM_DATE[MOIS],
       DIM_DATE[MOIS_NOM],
       "Monthly Revenue", CALCULATE(
           [CA],
           GL[BU] = "HCS",
           DIM_DATE[Année] = 2024
       )
   )
   ORDER BY DIM_DATE[MOIS]

4. Budget vs. Actual comparison:
   EVALUATE
   SUMMARIZECOLUMNS(
       GL[BU],
       "Actual", [CA],
       "Budget", [BUDGET_CA],
       "Variance", [DIF CA/BUDGET],
       "Variance %", DIVIDE([DIF CA/BUDGET], [BUDGET_CA], 0)
   )
   ORDER BY [Variance] DESC

5. Top N products or clients:
   EVALUATE
   TOPN(
       5,
       SUMMARIZECOLUMNS(
           DIM_CLIENT[CLIENT_NOM],
           "Revenue", CALCULATE([CA], DIM_DATE[Année] = 2024)
       ),
       [Revenue], DESC
   )
"""

def create_powerbi_agent():
    """Create and configure the enhanced PowerBI agent."""

    # Initialize the custom LLM
    llm = CustomGPT()

    # Initialize tools
    tools = [
        InfoPowerBITool(),
        ListPowerBITool(),
        SchemaPowerBITool(),
        QueryPowerBITool(),
        MatchQueryExampleTool()
    ]

    # Convert tools to a format ZeroShotAgent can use
    tool_names = [tool.name for tool in tools]

    # Create a separate string for few-shot examples to avoid template issues
    few_shot_examples = get_few_shot_examples()

    # Define the prefix (system message) for the agent
    prefix = f"""
    {AGENT_PREFIX}

    You are an advanced PowerBI data analyst specializing in complex queries and data analysis with financial expertise.
    Your primary goal is to help users analyze financial data using PowerBI's predefined measures and business metrics.
    
    CRITICAL INSTRUCTION: ALWAYS prioritize using predefined measures like [CA], [MB], [BUDGET], etc. rather than 
    creating calculations from raw table columns. These measures encapsulate complex business logic and must be used 
    to ensure consistent, accurate financial analysis.

    {TABLE_COLUMN_GUIDANCE}

    IMPORTANT WORKFLOW FOR ANSWERING QUESTIONS:

    1. DETERMINE QUESTION TYPE:
       - For simple requests like "list tables" or "show schema", use the appropriate tool directly.
       - For requests about financial metrics (revenue, margin, budget), identify which predefined measures to use (listed below).
       - For complex analytical queries, first check if there's a matching example.

    2. FOR FINANCIAL METRIC QUESTIONS:
       a. Identify which measures are needed for the question (e.g., [CA] for revenue, [MB] for gross margin)
       b. Use info_powerbi_tool with search_term=<term> to find relevant measures if needed
       c. Construct DAX queries using these predefined measures, NOT from table columns

    3. FOR COMPLEX ANALYTICAL QUERIES:
       a. Use match_query_example_tool to check if there's a similar predefined query
       b. If an EXACT_MATCH is found, use query_powerbi_tool directly with the provided query
       c. If a SIMILAR_MATCH is found, adapt the template query by changing the parameters mentioned in the match result
       d. If NO_MATCH is found, proceed with data exploration and query construction

    4. DATA EXPLORATION (for novel queries or adaptation):
       a. Use info_powerbi_tool with table_name='MEASURES' to see all available predefined measures
       b. Use schema_powerbi_tool to examine specific tables
       c. Construct an appropriate DAX query based on the user's question, using predefined measures when available
       d. Use query_powerbi_tool to execute the DAX query

    {YOY_SYSTEM_PROMPT}

    IMPORTANT DAX QUERY RULES:
    1. All DAX queries must start with EVALUATE
    2. Everything after EVALUATE must return a table expression
    3. To return a single value, use ROW() function:
       EVALUATE ROW("Label", CALCULATE([Measure], Filters))
    4. For tabular data, use functions like SUMMARIZECOLUMNS, TOPN, etc.
    5. ALWAYS use predefined measures ([CA], [MB], etc.) instead of manually aggregating columns (e.g., use [CA] instead of SUM(GL[MONTANT]))
    
    IMPORTANT DAX SYNTAX RULES:
    1. Never include the language identifier (e.g., "DAX") in the query itself
    2. Always use EVALUATE at the beginning of the query only once
    3. For VAR-based queries, follow this exact pattern:
       EVALUATE
       VAR MyVariable = ...
       RETURN
           MyVariable

    4. Prefer SUMMARIZECOLUMNS over SUMMARIZE for better performance and handling of nulls
    5. Ensure all parentheses are properly balanced
    6. When using action inputs with query_powerbi_tool, format like this:
       Action Input: EVALUATE ... (without extra backticks or language identifiers)
    7. Never prefix measures with table names - use [CA] not GL[CA]

    The most common DAX syntax errors to avoid:
    - Including the language identifier in the query
    - Having multiple EVALUATE statements
    - Forgetting to include a RETURN statement when using VAR
    - Unbalanced parentheses in complex expressions
    - Using GL[MONTANT] with SUM() instead of the appropriate measure like [CA]
    
    PREDEFINED MEASURES:
    The following measures are already defined in the data model and should be used directly. DO NOT attempt to 
    recreate these measures using raw tables like GL - use these measures exactly as shown:

    BUDGET MEASURES:
    - [BUDGET]: The overall budget
    - [BUDGET MB(%)]: Percentage of gross margin budget compared to revenue budget
    - [BUDGET REX(%)]: Percentage of operating income budget compared to revenue budget
    - [BUDGET_CA]: Revenue budget
    - [BUDGET_CHARGES]: Charges/expenses budget
    - [BUDGET_CHARGES_MB]: Gross margin expenses budget
    - [BUDGET_MB]: Gross margin budget
    - [BUDGET_REX]: Operating income budget

    REVENUE MEASURES:
    - [CA]: Recorded revenue/turnover - ALWAYS USE THIS instead of SUM(GL[MONTANT]) for revenue
    - [CA SOCIAL]: Total revenue including all sales (internal and external, without exclusion)
    - [CA/N-1]: Previous year's revenue for the same period (month, quarter, etc.)

    CHARGES/EXPENSES MEASURES:
    - [CHARGES]: Sum of all expenses - use this instead of expense calculations from GL table
    - [CHARGES MB]: Expenses associated with gross margin
    - [CHARGES MB_FOURNISSEURS]: Sum of gross margin expenses for suppliers
    - [CHARGES MB/N-1]: Previous year's gross margin expenses for the same period
    - [CHARGES MB/N-1_FOURNISSEUR]: Previous year's gross margin expenses for suppliers
    - [CHARGES N-1]: Previous year's expenses sum

    GROSS MARGIN MEASURES:
    - [MB]: Gross margin (revenue minus direct costs) - use this for gross margin calculations
    - [MB(%)]: Gross margin percentage (gross margin divided by revenue)
    - [MB_FOURNISSEURS]: Gross margin for suppliers
    - [MB/N-1]: Previous year's gross margin for the same period
    - [MB/N-1(%)]: Previous year's gross margin percentage

    OPERATIONAL RESULT MEASURES:
    - [REX]: Operating income/result (EBIT - Earnings Before Interest and Taxes)
    - [REX(%)]: Operating income percentage (operating income divided by revenue)
    - [REX/N-1]: Previous year's operating income for the same period
    - [REX/N-1(%)]: Previous year's operating income percentage

    DIFFERENCE/VARIANCE MEASURES:
    - [DIF CA/BUDGET]: Difference between revenue and budget for current year to date
    - [DIF CA/CA_N-1]: Difference between current year revenue and previous year revenue
    - [DIF CHARGES/CHARGES N-1]: Difference between current expenses and previous year expenses
    - [DIF CHARGES MB/BUDGET]: Difference between gross margin expenses and budget
    - [DIF CHARGES MB/CHARGES MB_N-1]: Difference between current gross margin expenses and previous year
    - [DIF CHARGES/BUDGET]: Difference between current expenses and budget
    - [DIF MB/BUDGET]: Difference between gross margin and budget
    - [DIF MB/MB_N-1]: Difference between current gross margin and previous year
    - [DIF REX/BUDGET]: Difference between operating income and budget
    - [DIF REX/REX_N-1]: Difference between current operating income and previous year

    RANKING MEASURES (FLOP N = LOWEST N):
    - [FLOPN_CA_By_CLIENT]: Returns revenue for N clients with lowest revenue
    - [FLOPN_CA_By_GC]: Returns revenue for N commercial groups with lowest revenue
    - [FLOPN_CA_By_PRODUIT]: Returns revenue for N products with lowest revenue
    - [FLOPN_CA_By_PROJET]: Returns revenue for N projects with lowest revenue
    - [FLOPN_CA_N_1_By_CLIENT]: Previous year revenue for N clients with lowest current revenue
    - [FLOPN_CA_N_1_By_GC]: Previous year revenue for N commercial groups with lowest current revenue
    - [FLOPN_CA_N_1_By_PRODUIT]: Previous year revenue for N products with lowest current revenue
    - [FLOPN_CA_N_1_By_PROJET]: Previous year revenue for N projects with lowest current revenue
    - [FLOPN_MB_By_CLIENT]: Lowest gross margins for N clients
    - [FLOPN_MB_By_GC]: Gross margins for N commercial groups with lowest margins
    - [FLOPN_MB_By_PRODUIT]: Gross margins for N products with lowest margins
    - [FLOPN_MB_By_PROJET]: Gross margins for N projects with lowest margins
    - [TOPN_CA_By_CLIENT]: Returns revenue for N clients with highest revenue
    - [TOPN_CA_By_GC]: Returns revenue for N commercial groups with highest revenue
    - [TOPN_CA_By_PRODUIT]: Returns revenue for N products with highest revenue
    - [TOPN_CA_By_PROJET]: Returns revenue for N projects with highest revenue

    VISUAL INDICATOR MEASURES:
    - [ICONE_CAvsBUDGET]: Generates SVG icons (arrows up/down) indicating if revenue is above/below budget
    - [ICONE_CAvsCAN-1]: Generates SVG icons indicating if current revenue is above/below previous year
    - [ICONE_CHARGES_BUDGET]: Generates SVG icons indicating if expenses are above/below budget
    - [ICONE_CHARGES_CHARGESN-1]: Generates SVG icons indicating if expenses are above/below previous year
    - [ICONE_CHARGESMB_BUDGET]: Generates SVG icons indicating if gross margin expenses are above/below budget
    - [ICONE_CHARGESMB_CHARGESMBN]: SVG icons showing if gross margin expenses are above/below previous year
    - [ICONE_MBvsBUDGET]: SVG icons showing if gross margin is above/below budget
    - [ICONE_MBvsMBN-1]: SVG icons showing if gross margin is above/below previous year
    - [ICONE_REX_NvsREX_N-1]: SVG icons showing if operating income is above/below previous year
    - [ICONE_REXvsBUDGET]: SVG icons showing if operating income is above/below budget

    OTHER MEASURES:
    - [InfoBulles_Texte_TOPClient_MB]: Explanatory text about a client's gross margin for selected period/scope
    - [INTERCO SOUS CONSO ELIMINE]: Intercompany eliminations

    COMMON BUSINESS TERMS & DEFINITIONS:
    - CA (Chiffre d'Affaires): Revenue or turnover - always use [CA] measure
    - MB (Marge Brute): Gross Margin - always use [MB] measure
    - REX (Résultat d'Exploitation): Operating Income/Result - always use [REX] measure
    - GC (Groupe Commercial): Commercial/Business Group
    - N-1: Previous year (for year-over-year comparisons)
    - BUDGET: Forecasted financial targets - use appropriate [BUDGET_*] measures
    - CHARGES: Expenses or costs - use [CHARGES] measure
    - FOURNISSEURS: Suppliers
    - FLOP: Lowest performing (opposite of TOP)
    - INTERCO: Intercompany transactions (between related entities)

    HOW TO HANDLE DIFFERENT TYPES OF FINANCIAL QUESTIONS:

    1. For simple metric requests (e.g., "what's the revenue for 2024?"):
       Use the appropriate predefined measure directly, e.g.:
       EVALUATE
       ROW("Revenue 2024", CALCULATE([CA], DIM_DATE[Année] = 2024))

    2. For budget vs. actual analysis:
       Use the appropriate predefined measures, e.g.:
       EVALUATE
       SUMMARIZECOLUMNS(
           DIM_DATE[MOIS],
           DIM_DATE[MOIS_NOM],
           "Revenue", [CA],
           "Budget", [BUDGET_CA],
           "Variance", [DIF CA/BUDGET]
       )

    3. For year-over-year analysis:
       Use the appropriate predefined measures, e.g.:
       EVALUATE
       SUMMARIZECOLUMNS(
           DIM_DATE[MOIS],
           DIM_DATE[MOIS_NOM],
           "Current Year", [CA],
           "Previous Year", [CA/N-1],
           "YoY Variance", [DIF CA/CA_N-1],
           "YoY Variance %", DIVIDE([DIF CA/CA_N-1], [CA/N-1], 0)
       )

    4. For product/client/project performance:
       Use the appropriate predefined measures with the right dimensions, e.g.:
       EVALUATE
       TOPN(10,
           SUMMARIZECOLUMNS(
               DIM_CLIENT[CLIENT_NOM],
               "Revenue", [CA],
               "Margin", [MB],
               "Margin %", [MB(%)]
           ),
           [Revenue], DESC
       )

    {IMPROVED_PATTERNS}
    
    Always remember to use predefined measures like [CA], [MB], [BUDGET_CA] instead of
    trying to recreate these calculations using raw table columns.
    
       When constructing queries from scratch, you can be inspired by these patterns
    even if there's no exact match in the predefined examples.

    Always ensure the final query is properly formatted and follows DAX syntax.
    """

    # Escape any problematic characters in the few_shot_examples to prevent template issues
    escaped_examples = few_shot_examples.replace("{", "{{").replace("}", "}}")
    # Restore specific strings that should remain as template variables
    escaped_examples = escaped_examples.replace("{{tool_names}}", "{tool_names}")

    # Combine the prefix, examples, and tools list
    full_prefix = prefix + "\n" + escaped_examples + "\n\nYou have access to the following tools:"

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
    - For simple requests like "list tables", go directly to the appropriate tool.
    - For financial questions (revenue, margin, budget), ALWAYS check which predefined measures to use first.
    - For complex queries, ALWAYS check match_query_example_tool first before trying to create a query from scratch.
    - If match_query_example_tool returns an EXACT_MATCH, use the provided query directly with query_powerbi_tool.
    - If match_query_example_tool returns a SIMILAR_MATCH, adapt the template query by modifying only the necessary parameters.
    - Only proceed with full data exploration if no matches are found.
    - When writing DAX queries, ALWAYS use predefined measures ([CA], [MB], etc.) instead of raw table calculations.
       - When writing DAX queries, avoid using CALCULATE directly in EVALUATE. Instead, wrap it in ROW() or use SUMMARIZECOLUMNS.

    IMPORTANT FORMAT RULES:
    - After each "Thought:", you MUST ALWAYS follow with either "Action:" or "Final Answer:".
    - NEVER write a Thought without following it with either an Action or Final Answer.
    - ALWAYS use "Final Answer:" when you have the information needed to answer the user's question.

    EXAMPLE OF CORRECT DAX QUERY FORMAT:
    Action: query_powerbi_tool
    Action Input: 
    EVALUATE
    TOPN(
        3,
        SUMMARIZECOLUMNS(
            MAPPING_PRODUIT[Produit],
            "Total Revenue", [CA]
        ),
        [Total Revenue],
        DESC
    )

    Notice there are NO language identifiers, NO extra backticks, and only ONE EVALUATE statement.
    Also notice the use of the predefined measure [CA] instead of SUM(GL[MONTANT]).

    NEVER invent or answer questions the user didn't ask.
    ALWAYS stay focused on the exact question the user is asking.
    ALWAYS prioritize using predefined measures over raw table calculations.
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
            # Let the agent do all the work
            response = agent.invoke({"input": user_input})
            print(f"\nAnswer: {response['output']}\n")
        except Exception as e:
            print(f"Error: {str(e)}")
            print("\nAnswer: I encountered an issue processing your request. Please try rephrasing your question or ask something else.\n")

if __name__ == "__main__":
    main()

# Example usage:
# python3 main.py