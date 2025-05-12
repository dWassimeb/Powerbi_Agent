"""
Comprehensive metadata for PowerBI dataset tables with table types.
"""

DATASET_METADATA = {
    "DIM_CLIENT": {
        "description": "Dimension table containing all client information and attributes",
        "type": "dimension",
        "columns": {
            "CLIENT_ID": "Unique identifier for each client",
            "CLIENT_NOM": "Client name",
            "CLIENT_TYPE": "Type of client (B2B, B2C, etc.)",
            "CLIENT_SEGMENT": "Market segment the client belongs to",
            "CLIENT_PAYS": "Country where the client is located"
        }
    },
    "DIM_DATE": {
        "description": "Time dimension table with date hierarchies",
        "type": "dimension",
        "columns": {
            "DATE_ID": "Unique identifier for each date (primary key)",
            "DATE": "Calendar date in format YYYY-MM-DD",
            "ANNEE": "Year number",
            "MOIS": "Month number (1-12)",
            "TRIMESTRE": "Quarter number (1-4)",
            "SEMESTRE": "Semester number (1-2)",
            "JOUR_SEMAINE": "Day of week (1-7)",
            "EST_JOUR_FERIE": "Boolean indicating if date is a holiday"
        }
    },
    "DIM_SOCIETE": {
        "description": "Dimension table containing company information and hierarchies",
        "type": "dimension",
        "columns": {
            "ID_SOCIETE_JURIDIQUE": "Unique identifier for each legal company entity (primary key)",
            "Lib Société Juridique": "Legal company name",
            "X_INSERT_DATE": "Date when record was inserted",
            "X_UPDATE_DATE": "Date when record was last updated",
            "X_STATUS": "Status of the record",
            "Lib Société Juridique Court": "Short legal company name",
            "X_START_DATE": "Start date for the record validity",
            "X_END_DATE": "End date for the record validity",
            "COMPTA": "Accounting identifier",
            "DATE_FIN_COMPTA": "End date for accounting period",
            "Société": "Business entity name",
            "Code Société": "Business entity code"
        }
    },
    "GL": {
        "description": "Main fact table containing financial transactions and analytical account amounts",
        "type": "fact",
        "columns": {
            "EXERCICE": "Fiscal year",
            "MOIS": "Month number (1-12)",
            "Date Comptable": "Accounting date for the transaction",
            "SOCIETE": "Company code",
            "Num_Tiers": "Third-party identifier",
            "CDR": "Cost center code",
            "CDR_LIB": "Cost center name",
            "ACTIVITE": "Activity code",
            "ACTIVITE_LIB": "Activity name",
            "PROJET": "Project code",
            "PROJET_LIB": "Project name",
            "PRODUIT": "Product code",
            "PRODUIT_LIB": "Product name",
            "Client": "Client code",
            "Client_Lib": "Client name",
            "COMPTE_ANALYTIQUE": "Analytical account code",
            "COMPTE_LIBELLE": "Analytical account name",
            "MONTANT": "Financial amount (revenue or cost)",
            "Contrepartie": "Counterparty",
            "Raison_Sociale": "Company legal name",
            "Siret_Tiers": "SIRET number of third-party",
            "G_Docaposte": "Docaposte group flag",
            "G_Softeam": "Softeam group flag",
            "Perimetre": "Business perimeter",
            "Nom_Fichier": "Source file name",
            "CONSO": "Consolidation flag",
            "PROJET_CDR_KEY": "Composite key linking project and cost center",
            "SOCIETE_CDR_KEY": "Composite key linking company and cost center",
            "CPTANAL_CDR_KEY": "Composite key linking analytical account and cost center",
            "CPT_ANAL_KEY": "Analytical account key",
            "EXERCICE_PROJECT_KEY": "Composite key linking fiscal year and project",
            "SOCIETE_ACTIVITE_KEY": "Composite key linking company and activity",
            "SOCIETE_PRODUIT_KEY": "Composite key linking company and product",
            "Société": "Company name",
            "Pôle": "Business pole",
            "BU": "Business Unit",
            "Sous BU": "Sub Business Unit",
            "Marché": "Market",
            "Groupe Client": "Client group",
            "Nature de Charges": "Cost type"
        }
    },
    "MAPPING_ACTIVITE": {
        "description": "Dimension table mapping activities by company and fiscal year",
        "type": "dimension",
        "columns": {
            "EXERCICE": "Fiscal year",
            "Activité": "Activity code",
            "Description": "Activity description",
            "Niv1_Activité": "Activity hierarchy level 1",
            "Niv2_Activité": "Activity hierarchy level 2",
            "Niv3_Activité": "Activity hierarchy level 3",
            "Niv4_Activité": "Activity hierarchy level 4",
            "Niv5_Activité": "Activity hierarchy level 5",
            "Code Société": "Company code",
            "SOCIETE_ACTIVITE_KEY": "Composite key linking company and activity"
        }
    },
    "MAPPING_CDR": {
        "description": "Dimension table mapping cost centers",
        "type": "dimension",
        "columns": {
            "CDR_CODE": "Cost center code",
            "CDR_NAME": "Cost center name",
            "CDR_LEVEL1": "Cost center hierarchy level 1",
            "CDR_LEVEL2": "Cost center hierarchy level 2",
            "CDR_LEVEL3": "Cost center hierarchy level 3",
            "ACTIVE": "Flag indicating if cost center is active",
            "SOCIETE_ID": "Company identifier that owns the cost center"
        }
    },
    "MAPPING_COMPTE": {
        "description": "Dimension table mapping analytical accounts",
        "type": "dimension",
        "columns": {
            "COMPTE_ID": "Analytical account identifier",
            "COMPTE_CODE": "Analytical account code",
            "COMPTE_NAME": "Analytical account name",
            "NIV1_COMPTE": "Account hierarchy level 1",
            "NIV2_COMPTE": "Account hierarchy level 2",
            "NIV3_COMPTE": "Account hierarchy level 3",
            "NIV4_COMPTE": "Account hierarchy level 4",
            "NIV5_COMPTE": "Account hierarchy level 5",
            "ACCOUNT_TYPE": "Type of account (expense, revenue, etc.)",
            "IS_ACTIVE": "Flag indicating if account is active"
        }
    },
    "MAPPING_PRODUIT": {
        "description": "Dimension table mapping products by company and fiscal year",
        "type": "dimension",
        "columns": {
            "EXERCICE": "Fiscal year",
            "Code Produit": "Product code",
            "Produit": "Product name",
            "Niv1_CRM": "Product hierarchy level 1 in CRM",
            "Niv2_CRM": "Product hierarchy level 2 in CRM",
            "Niv3_CRM": "Product hierarchy level 3 in CRM",
            "Niv4_CRM": "Product hierarchy level 4 in CRM",
            "Niv5_CRM": "Product hierarchy level 5 in CRM",
            "BU PROD": "Business Unit producing the product",
            "REF_CAT_CRM": "CRM category reference",
            "CODE SOCIETE": "Company code",
            "LIB_SOC_REG": "Regional company name",
            "SOCIETE_PRODUIT_KEY": "Composite key linking company and product",
            "CODE_NOM_PRODUIT": "Composite code of product code and name"
        }
    },
    "MAPPING_PROJET_CDR": {
        "description": "Mapping table linking projects and cost centers",
        "type": "dimension",
        "columns": {
            "PROJET_ID": "Project identifier",
            "PROJET_CODE": "Project code",
            "PROJET_NAME": "Project name",
            "CDR_ID": "Cost center identifier",
            "CDR_CODE": "Cost center code",
            "PROJET_CDR_CODE": "Composite code of project and cost center",
            "IS_ACTIVE": "Flag indicating if mapping is active",
            "START_DATE": "Start date of the mapping",
            "END_DATE": "End date of the mapping"
        }
    },
    "MAPPING_PROJECT_MULTIBU": {
        "description": "Mapping table for projects spanning multiple business units",
        "type": "dimension",
        "columns": {
            "PROJECT_ID": "Project identifier",
            "PROJECT_CODE": "Project code",
            "PROJECT_NAME": "Project name",
            "BU_ID": "Business Unit identifier",
            "BU_CODE": "Business Unit code",
            "BU_NAME": "Business Unit name",
            "ALLOCATION_PCT": "Percentage allocation to the BU",
            "START_DATE": "Start date of the allocation",
            "END_DATE": "End date of the allocation"
        }
    },
    "INDICATEURS": {
        "description": "Table containing various business indicators and metrics",
        "type": "fact",
        "columns": {
            "INDICATOR_ID": "Unique identifier for the indicator",
            "INDICATOR_NAME": "Name of the indicator",
            "INDICATOR_VALUE": "Value of the indicator",
            "PERIOD": "Time period for the indicator",
            "ENTITY_ID": "Entity the indicator relates to",
            "ENTITY_TYPE": "Type of entity (Project, Company, etc.)",
            "UPDATE_DATE": "Date of last update",
            "SOURCE": "Source of the indicator data"
        }
    }
}

# Define the relationships between tables in the star schema
RELATIONSHIPS = [
    {"from_table": "GL", "from_key": "Client", "to_table": "DIM_CLIENT", "to_key": "CLIENT_ID",
     "description": "Links transactions to clients"},

    {"from_table": "GL", "from_key": "SOCIETE", "to_table": "DIM_SOCIETE", "to_key": "Code Société",
     "description": "Links transactions to companies"},

    {"from_table": "GL", "from_key": "Date Comptable", "to_table": "DIM_DATE", "to_key": "DATE",
     "description": "Links transactions to dates"},

    {"from_table": "GL", "from_key": "COMPTE_ANALYTIQUE", "to_table": "MAPPING_COMPTE", "to_key": "COMPTE_CODE",
     "description": "Links transactions to analytical accounts"},

    {"from_table": "GL", "from_key": "PROJET", "to_table": "MAPPING_PROJET_CDR", "to_key": "PROJET_CODE",
     "description": "Links transactions to projects"},

    {"from_table": "GL", "from_key": "PRODUIT", "to_table": "MAPPING_PRODUIT", "to_key": "Code Produit",
     "description": "Links transactions to products"},

    {"from_table": "GL", "from_key": "ACTIVITE", "to_table": "MAPPING_ACTIVITE", "to_key": "Activité",
     "description": "Links transactions to activities"},

    {"from_table": "GL", "from_key": "CDR", "to_table": "MAPPING_CDR", "to_key": "CDR_CODE",
     "description": "Links transactions to cost centers"},

    {"from_table": "MAPPING_PROJECT_MULTIBU", "from_key": "BU_ID", "to_table": "DIM_SOCIETE", "to_key": "ID_SOCIETE_JURIDIQUE",
     "description": "Links projects to business units"},

    {"from_table": "MAPPING_PROJET_CDR", "from_key": "PROJET_CDR_CODE", "to_table": "MAPPING_CDR", "to_key": "CDR_CODE",
     "description": "Links project-cost center mappings to cost centers"},

    {"from_table": "MAPPING_PRODUIT", "from_key": "CODE SOCIETE", "to_table": "DIM_SOCIETE", "to_key": "Code Société",
     "description": "Links products to companies"},

    {"from_table": "MAPPING_ACTIVITE", "from_key": "Code Société", "to_table": "DIM_SOCIETE", "to_key": "Code Société",
     "description": "Links activities to companies"}
]