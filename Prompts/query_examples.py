"""
Example queries for the PowerBI LLM to help it understand how to formulate complex queries.
"""
# Example DAX queries
DAX_EXAMPLES = [

    {
        "question": "give me the total revenu of the product P231 for the year 2024?",
        "query": """EVALUATE
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
)"""
    },





    {
        "question": "Evolution du CA mensuel sur T1 et T2 2024 du client Anah de la sous BU Back Office",
        "query": """EVALUATE
SUMMARIZECOLUMNS(
    DIM_DATE[Mois],
    DIM_DATE[MOIS_NOM],
    "Chiffre d'Affaires", 
    CALCULATE(
        [CA],
        DIM_CLIENT[RAISON_SOCIALE_DO] = "Annah",
        GL[Sous BU] = "Back office",
        DIM_DATE[Année] = 2024,
        DIM_DATE[Mois] >= 1,
        DIM_DATE[Mois] <= 6
    )
)
ORDER BY DIM_DATE[Mois]"""
    },

    {
        "question": "Marge brute de la sous BU Manufacture pour le mois de septembre 2024",
        "query": """EVALUATE
ROW(
    "Gross Margin", 
    CALCULATE(
        [MB],
        GL[Sous BU] = "Manufacture",
        DIM_DATE[Année] = 2024,
        DIM_DATE[Mois] = 9
    )
)"""
    },

    {
        "question": "What is the total revenue by product for Acme Corp in 2023?",
        "query": """EVALUATE
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
ORDER BY [Total Revenue] DESC"""
    },

    {
        "question": "give me the best selling three products in termes of revenu, in the year 2024",
        "query": """EVALUATE
TOPN(
    3,
    SUMMARIZECOLUMNS(
        MAPPING_PRODUIT[Produit],
        "Total Revenue", 
        CALCULATE(
            SUM(GL[MONTANT]),
            GL[EXERCICE] = 2024
        )
    ),
    [Total Revenue],
    DESC
)"""
    },

    {
        "question": "Show me monthly expenses by cost center for project PRJ2023-001",
        "query": """EVALUATE
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
ORDER BY DIM_DATE[MOIS], [Total Expenses] DESC"""
    },

    {
        "question": "Compare revenue across product categories and business units, with year-over-year growth",
        "query": """EVALUATE
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
ORDER BY [YoY Growth] DESC"""
    }
]