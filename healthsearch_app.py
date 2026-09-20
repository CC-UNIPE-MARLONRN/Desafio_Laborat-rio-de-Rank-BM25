import re
import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


st.set_page_config(
    page_title="HealthSearch",
    page_icon="❤",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root { --wine: #ff6b78; --coral: #ff9b8f; --ink: #edf2f7; --mist: #17212b; }
    .stApp { background: radial-gradient(circle at 12% 8%, #263b4b 0%, transparent 32%),
        linear-gradient(135deg, #0e151c 0%, #17212b 55%, #211b2a 100%); color: #edf2f7; }
    [data-testid="stSidebar"] { background: #111a23; border-right: 1px solid #344454; }
    h1 { color: var(--coral); letter-spacing: 0; }
    h2, h3, p, label { color: var(--ink); letter-spacing: 0; }
    .heart-header { display: flex; gap: 14px; align-items: center; padding: 18px 22px;
        border-left: 7px solid var(--coral); background: rgba(22,32,43,.88);
        box-shadow: 0 8px 24px rgba(0,0,0,.24); margin-bottom: 20px; }
    .heart-symbol { color: var(--coral); font-size: 2.4rem; line-height: 1; }
    .heart-subtitle { color: #52606d; margin: 2px 0 0; }
    div[data-baseweb="tab-list"] { gap: 6px; }
    button[data-baseweb="tab"] { color: var(--coral); }
    </style>
    """,
    unsafe_allow_html=True,
)


DOCUMENTS = [
    {
        "id": "Doc 1",
        "titulo": "Protocolo Emergência ECG",
        "texto": "Pacientes com dor precordial aguda e suspeita de síndrome coronariana devem realizar eletrocardiograma CÓD-ECG-12D em até 10 minutos.",
    },
    {
        "id": "Doc 2",
        "titulo": "Guia de Farmacologia Cardíaca",
        "texto": "O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários reduz a mortalidade no infarto agudo do miocárdio.",
    },
    {
        "id": "Doc 3",
        "titulo": "Diretriz de Hipertensão Arterial",
        "texto": "A crise hipertensiva severa requer administração de anti-hipertensivos venosos e monitoramento contínuo da pressão arterial na UTI.",
    },
    {
        "id": "Doc 4",
        "titulo": "Manual de AVC Isquêmico",
        "texto": "O acidente vascular cerebral isquêmico agudo deve ser tratado com trombolíticos venosos em até quatro horas e meia do início dos sintomas.",
    },
    {
        "id": "Doc 5",
        "titulo": "Protocolo de Reanimação RCR",
        "texto": "Parada cardiorrespiratória em adultos exige compressões torácicas contínuas de alta qualidade e desfibrilação precoce no código azul.",
    },
    {
        "id": "Doc 6",
        "titulo": "Procedimentos de UTI Geral",
        "texto": "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas, recomenda-se a monitorização cardíaca contínua por telemetria.",
    },
]


@st.cache_resource
def get_stopwords() -> set[str]:
    try:
        return set(stopwords.words("portuguese"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("portuguese"))


def tokenize(text: str) -> list[str]:
    text = re.sub(r"[^\w\s-]", " ", text.lower(), flags=re.UNICODE)
    return [token for token in text.split() if token not in get_stopwords()]


@st.cache_resource(show_spinner="Carregando modelo de embeddings...")
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner="Carregando cross-encoder...")
def load_cross_encoder_model() -> CrossEncoder:
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def cosine_similarity(query_vector: np.ndarray, document_vectors: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vector)
    document_norms = np.linalg.norm(document_vectors, axis=1)
    return (document_vectors @ query_vector) / (document_norms * query_norm)


st.markdown(
    '<div class="heart-header"><div class="heart-symbol">❤</div><div>'
    '<h1>HealthSearch</h1><p class="heart-subtitle">Inteligência para encontrar a conduta clínica certa no momento certo.</p>'
    '</div></div>',
    unsafe_allow_html=True,
)

st.sidebar.header("Parâmetros BM25")
k1 = st.sidebar.slider("Saturação de frequência (k1)", 0.0, 3.0, 1.2, 0.1)
b = st.sidebar.slider("Normalização por comprimento (b)", 0.0, 1.0, 0.75, 0.05)

st.sidebar.header("Fusão RRF")
alpha = st.sidebar.slider("Peso BM25 (alpha)", 0.0, 1.0, 0.5, 0.05)
rrf_k = 60
st.sidebar.write(f"Constante k_RRF: {rrf_k}")

st.sidebar.header("Bônus: Cross-Encoder")
use_cross_encoder = st.sidebar.checkbox("Ativar re-ranking dos Top-3", value=False)

query = st.text_input(
    "Digite sua consulta",
    value="ataque cardíaco",
    placeholder="Ex.: infarto, ECG-12D ou pressão arterial",
)

query_started_at = time.perf_counter()
tokenized_documents = [tokenize(document["texto"]) for document in DOCUMENTS]
bm25 = BM25Okapi(tokenized_documents, k1=k1, b=b)
query_tokens = tokenize(query)

if not query_tokens:
    st.warning("Digite uma consulta com termos válidos.")
    st.stop()

bm25_scores = np.asarray(bm25.get_scores(query_tokens), dtype=float)
model = load_embedding_model()
document_embeddings = model.encode(
    [document["texto"] for document in DOCUMENTS],
    convert_to_numpy=True,
    normalize_embeddings=True,
)
query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
semantic_scores = cosine_similarity(query_embedding, document_embeddings)

bm25_order = np.argsort(-bm25_scores)
semantic_order = np.argsort(-semantic_scores)
bm25_ranks = {index: rank for rank, index in enumerate(bm25_order, start=1)}
semantic_ranks = {index: rank for rank, index in enumerate(semantic_order, start=1)}

results: list[dict[str, Any]] = []
for index, document in enumerate(DOCUMENTS):
    rank_bm25 = bm25_ranks[index]
    rank_semantic = semantic_ranks[index]
    rrf_score = alpha * (1 / (rrf_k + rank_bm25)) + (1 - alpha) * (1 / (rrf_k + rank_semantic))
    results.append(
        {
            "Documento": document["id"],
            "Título": document["titulo"],
            "Trecho Clínico": document["texto"],
            "Score BM25": bm25_scores[index],
            "Similaridade de Cosseno": semantic_scores[index],
            "Rank BM25": rank_bm25,
            "Rank Semântico": rank_semantic,
            "Score RRF": rrf_score,
        }
    )

results_df = pd.DataFrame(results)
hybrid_df = results_df.sort_values("Score RRF", ascending=False).reset_index(drop=True)
hybrid_df.index += 1

reranked_df = hybrid_df.head(3).copy()
if use_cross_encoder:
    cross_encoder = load_cross_encoder_model()
    pairs = [[query, text] for text in reranked_df["Trecho Clínico"]]
    reranked_df["Score Cross-Encoder"] = cross_encoder.predict(pairs)
    reranked_df = reranked_df.sort_values("Score Cross-Encoder", ascending=False)

query_time_ms = (time.perf_counter() - query_started_at) * 1000
st.caption(
    f"Métricas da consulta: {query_time_ms:.2f} ms | "
    f"{len(results_df)} documentos avaliados | "
    f"{len(reranked_df)} candidatos no resultado final"
)

tab_matrix, tab_lexical, tab_semantic, tab_hybrid, tab_rerank = st.tabs(
    ["Matriz Comparativa", "Busca Léxica (BM25)", "Busca Semântica", "Busca Híbrida (RRF)", "Cross-Encoder"]
)

with tab_matrix:
    st.subheader("Comparação dos Rankings")
    st.dataframe(
        results_df[
            ["Documento", "Título", "Rank BM25", "Rank Semântico", "Score RRF"]
        ].sort_values("Score RRF", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

with tab_lexical:
    st.subheader(f"Ranking BM25 (k1={k1}, b={b})")
    st.dataframe(
        results_df.sort_values("Score BM25", ascending=False)[
            ["Documento", "Título", "Score BM25", "Trecho Clínico"]
        ],
        use_container_width=True,
        hide_index=True,
    )

with tab_semantic:
    st.subheader("Ranking por Similaridade de Cosseno")
    st.dataframe(
        results_df.sort_values("Similaridade de Cosseno", ascending=False)[
            ["Documento", "Título", "Similaridade de Cosseno", "Trecho Clínico"]
        ],
        use_container_width=True,
        hide_index=True,
    )

with tab_hybrid:
    st.subheader(f"Ranking Híbrido RRF (alpha={alpha}, k_RRF={rrf_k})")
    st.dataframe(
        hybrid_df[
            [
                "Documento",
                "Título",
                "Score RRF",
                "Score BM25",
                "Similaridade de Cosseno",
                "Trecho Clínico",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

with tab_rerank:
    st.subheader("Re-ranking dos Top-3 candidatos do RRF")
    if not use_cross_encoder:
        st.info("Ative o checkbox do Cross-Encoder na barra lateral para comparar a nova ordem de relevância.")
    else:
        st.dataframe(
            reranked_df[
                ["Documento", "Título", "Score Cross-Encoder", "Score RRF", "Trecho Clínico"]
            ],
            use_container_width=True,
            hide_index=True,
        )