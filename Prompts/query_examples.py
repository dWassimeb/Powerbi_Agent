"""
Example queries for the PowerBI LLM to help it understand how to formulate complex queries.
"""
from Agent_LangGraph.Prompts.query_examples import DAX_EXAMPLES

# Example DAX queries
DAX_EXAMPLES = [
    {
        "question": "give me the total revenu of the product P231 for the year 2024?",
        "query": """
EVALUATE
SUMMARIZECOLUMNS(
    MAPPING_PRODUIT[Code Produit],
    FILTER(
        VALUES(GL[EXERCICE]),
        GL[EXERCICE] = 2024
    ),
    FILTER(
        VALUES(GL[PRODUIT]),
        GL[PRODUIT] = "P231"
    ),
    "Total Revenue", 
    SUM(GL[MONTANT])
)
"""
    },

    {
        "question": "TOP 3 produits par MB de la sous BU Digital Solutions en 2024",
        "query": """
"	VAR top3Tavle =
	TOPN(
		3,
		SUMMARIZE(
			MAPPING_PRODUIT,
			MAPPING_PRODUIT[Produit],
			""Marge Brute"", CALCULATE(
				[MB],
				GL[Sous BU] = ""Digital Solutions"",
				DIM_DATE[Année] = 2024
			)
		),
		[Marge Brute],
		DESC
	)
	VAR TOP3TRIEE = ADDCOLUMNS(
		top3Tavle,
		""Rank Marge Brute"", RANKX(
			top3Tavle,
			[Marge Brute],
			,
			DESC,
			Dense
		)
	)
	VAR tableFinalTriee = FILTER(
		TOP3TRIEE,
		[Rank Marge Brute] <= 3
	)

	RETURN
		tableFinalTriee"
"""
    },

    {
        "question": "What is the total revenue by product for Acme Corp in 2023?",
        "query": """
EVALUATE
SUMMARIZECOLUMNS(
    MAPPING_PRODUIT[Produit],
    FILTER(
        VALUES(DIM_DATE[ANNEE]),
        DIM_DATE[ANNEE] = 2023
    ),
    FILTER(
        VALUES(DIM_CLIENT[CLIENT_NOM]),
        DIM_CLIENT[CLIENT_NOM] = "Acme Corp"
    ),
    "Total Revenue", 
    CALCULATE(
        SUM(GL[MONTANT]),
        FILTER(
            GL,
            GL[COMPTE_ANALYTIQUE] IN {"7001", "7002", "7003"} // Revenue account codes
        )
    )
)
ORDER BY [Total Revenue] DESC
"""
    },

    {
        "question": "Show me monthly expenses by cost center for project PRJ2023-001",
        "query": """
EVALUATE
SUMMARIZECOLUMNS(
    DIM_DATE[MOIS],
    MAPPING_CDR[CDR_NAME],
    FILTER(
        VALUES(GL[PROJET]),
        GL[PROJET] = "PRJ2023-001"
    ),
    "Total Expenses", 
    CALCULATE(
        SUM(GL[MONTANT]),
        FILTER(
            GL,
            GL[COMPTE_ANALYTIQUE] IN {"6001", "6002", "6003"} // Expense account codes
        )
    )
)
ORDER BY DIM_DATE[MOIS], [Total Expenses] DESC
"""
    },

    {
        "question": "Compare revenue across product categories and business units, with year-over-year growth",
        "query": """
EVALUATE
SUMMARIZECOLUMNS(
    MAPPING_PRODUIT[Niv1_CRM], // Product category
    DIM_SOCIETE[BU], // Business unit
    "Revenue 2022", 
    CALCULATE(
        SUM(GL[MONTANT]),
        FILTER(ALL(DIM_DATE), DIM_DATE[ANNEE] = 2022),
        FILTER(GL, GL[COMPTE_ANALYTIQUE] IN {"7001", "7002", "7003"}) // Revenue accounts
    ),
    "Revenue 2023", 
    CALCULATE(
        SUM(GL[MONTANT]),
        FILTER(ALL(DIM_DATE), DIM_DATE[ANNEE] = 2023),
        FILTER(GL, GL[COMPTE_ANALYTIQUE] IN {"7001", "7002", "7003"}) // Revenue accounts
    ),
    "YoY Growth", 
    VAR Rev2022 = CALCULATE(
        SUM(GL[MONTANT]),
        FILTER(ALL(DIM_DATE), DIM_DATE[ANNEE] = 2022),
        FILTER(GL, GL[COMPTE_ANALYTIQUE] IN {"7001", "7002", "7003"})
    )
    VAR Rev2023 = CALCULATE(
        SUM(GL[MONTANT]),
        FILTER(ALL(DIM_DATE), DIM_DATE[ANNEE] = 2023),
        FILTER(GL, GL[COMPTE_ANALYTIQUE] IN {"7001", "7002", "7003"})
    )
    RETURN DIVIDE(Rev2023 - Rev2022, Rev2022, 0)
)
ORDER BY [YoY Growth] DESC
"""
    }
]








