# Before running, install required packages:
# pip install -r requirements.txt

import itertools
import pandas as pd
from uuid import uuid4
import streamlit as st
from operator import itemgetter

from model import CustomGPT
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_core.prompts import ChatPromptTemplate

from rank_bm25 import BM25Okapi
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings, HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ParentDocumentRetriever, MultiQueryRetriever, EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser
import chromadb
from chromadb.config import Settings
from langchain.schema.runnable import RunnableMap
from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun


# ----------------------------- Prompt Parameters -----------------------------
system_template = (
    "{context}"
)
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)


##########################################################################################
#################################  EMBEDDING  ############################################
##########################################################################################

@st.cache_resource
def load_emb_model(model="hf"):
    """
    @param model: le modèle d'embedding à utiliser : "hf" pour hugging face, "gpt4all" pour gpt4all
    @return: Modèle d'embedding
    """
    embeddings = None

    if model == "HuggingFace":
        embeddings = HuggingFaceEmbeddings()
    if model == "GPT4All":
        embeddings = GPT4AllEmbeddings()

    return embeddings


##########################################################################################
#################################  DATA PREP  ############################################
##########################################################################################

def load_doc(file_name="all_fiches_data.xlsx", index_folder="data/"):
    file = pd.read_excel(index_folder + file_name)
    return file


def to_doc(text_df: pd.DataFrame, text_col: str = 'text') -> list[Document]:
    """
    Transformer les documents en entrée en objet Document

    @param text_df: DataFrame with the text data and metadata
    @param text_col: String with name of column with text data
    @return: List of Document objects containing text data & metadata
    """
    text_df["id"] = text_df.index.values
    rec = text_df.to_dict("records")
    meta = [{k: v for k, v in r.items() if k != "c"} for r in rec]
    text = [r[text_col] for r in rec]
    doc = [Document(page_content=t, metadata=m) for t, m in zip(text, meta)]
    return doc


##########################################################################################
#################################  RETRIEVERS  ###########################################
##########################################################################################


class CustomMultiQuery:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.multi_query_retriever = MultiQueryRetriever.from_llm(retriever=self.retriever.get_retriever(),
                                                                  llm=self.llm)
        self.name = "MultiQuery_" + retriever.name

    def get_docs_w_scores(self, query):
        run_manager = AsyncCallbackManagerForChainRun(run_id=1, handlers=[], inheritable_handlers=[])
        queries = [query] + self.multi_query_retriever.generate_queries(query, run_manager)
        return list(itertools.chain.from_iterable([self.retriever.get_docs_w_scores(q) for q in queries]))

    def get_retriever(self):
        multi_query_retriever = MultiQueryRetriever.from_llm(retriever=self.retriever.get_retriever(), llm=self.llm)
        multi_query_retriever.name = "MultiQuery_" + self.retriever.name
        return multi_query_retriever


class CustomChroma:
    def __init__(self, ref_data, column, emb_model, n_docs, collection_name, client):
        doc = to_doc(ref_data, column)
        self.name = "ChromaRetriever"
        self.n_docs = n_docs
        _ = client.get_or_create_collection(collection_name)

        self.vectorStore = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=emb_model,
        )
        self.vectorStore.add_documents(ids=[str(uuid4()) for _ in range(len(doc))], documents=doc)

    def get_docs_w_scores(self, query):
        return sorted(self.vectorStore.similarity_search_with_score(query, k=self.n_docs), key=lambda x: x[1],
                      reverse=False)

    def get_retriever(self):
        chroma_retriever = self.vectorStore.as_retriever(search_kwargs={"k": self.n_docs})
        chroma_retriever.name = self.name
        return chroma_retriever


class CustomBM25:
    def __init__(self, ref_data, column, n_docs):
        docs = ref_data[column].values
        self.name = "BM25Retriever"
        self.docs = to_doc(ref_data, column)
        self.n_docs = n_docs
        self.tokenized_corpus = [doc.split(" ") for doc in docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def get_docs_w_scores(self, query):
        tokenized_query = query.split(" ")
        doc_scores = zip(self.docs, self.bm25.get_scores(tokenized_query))
        return sorted(doc_scores, key=lambda x: x[1], reverse=True)[:self.n_docs]

    def get_retriever(self):
        bm25_retriever = BM25Retriever.from_documents(self.docs)
        bm25_retriever.k = self.n_docs
        bm25_retriever.name = self.name
        return bm25_retriever


class CustomEnsemble:
    def __init__(self, retrievers_list, weights):
        self.name = "Ensemble_" + "_".join([r.name for r in retrievers_list])
        self.retrievers_list = retrievers_list
        self.weights = weights
        self.retrievers = zip(retrievers_list, weights)
        self.ensemble_retriever = EnsembleRetriever(retrievers=self.retrievers_list, weights=self.weights)

    def get_docs_w_scores(self, query):
        result = self.ensemble_retriever.invoke(query)
        ids_ensemble = [res.metadata["id"] for res in result]
        res_dict = {}
        for r in self.retrievers_list:
            result_tmp = r.invoke(query)
            ids_tmp = [res.metadata["id"] for res in result_tmp]
            common_ids = set(ids_ensemble) & set(ids_tmp)
            res_dict[r.name] = common_ids
        final_result = []
        for i, res in enumerate(result):
            a = res.copy()
            a.metadata["retrievers"] = []
            for ret, c_ids in res_dict.items():
                if a.metadata["id"] in c_ids:
                    a.metadata["retrievers"].append(ret)
            final_result.append((a, a.metadata["retrievers"]))
        return final_result


def get_retriever(name, data_file, column, embedding, n_docs, multi_query, collection_name):
    if name == "BM25Retriever":
        retriever = CustomBM25(data_file, column, n_docs)
        if multi_query:
            retriever = CustomMultiQuery(retriever, CustomGPT())
        return retriever
    elif name == "ChromaRetriever":
        embedding_ = load_emb_model(embedding)
        client = chromadb.PersistentClient(path=st.session_state.chroma_path, settings=Settings(allow_reset=True))
        retriever = CustomChroma(data_file, column, embedding_, n_docs, collection_name, client)
        if multi_query:
            retriever = CustomMultiQuery(retriever, CustomGPT())
        return retriever
    elif name == "MultiQuery":
        return
    elif name == "ParentRetriever":
        return
    else:
        return


##########################################################################################
######################################  CHAIN  ###########################################
##########################################################################################

def build_chain(prompt, llm):
    retrieve_docs_chain = RunnableMap(
        {
            "documents": lambda x: x["retriever"].get_docs_w_scores(x["question"]),
            "question": itemgetter("question")
        }
    )

    get_position = RunnableMap(
        {
            "position": lambda x: [{"document": d[0].page_content, "position": i, "score": d[1]} for i, d in
                                   enumerate(x["documents"])],
            "question": itemgetter("question")
        }
    )

    concat_docs = RunnableMap(
        {
            "context": lambda x: "\n\n".join([f"Article {d['position']} :\n {d['document']}" for d in x["position"]]),
            "question": itemgetter("question"),
            "documents": itemgetter("position")
        }
    )

    prompt_chain = RunnableMap(
        {
            "answer": prompt | llm | StrOutputParser(),
            "documents": itemgetter("documents")
        }
    )

    retrievers_chain = (retrieve_docs_chain | get_position | concat_docs | prompt_chain)

    return retrievers_chain


def get_model_chain():
    m = CustomGPT()
    chain = build_chain(prompt, m)
    return chain
