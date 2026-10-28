from __future__ import annotations
import os
import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the project .env file so OPENAI_API_KEY is available to this app.
load_dotenv()

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

st.set_page_config(
    page_title="RelevanceIQ | Patent Search",
    page_icon="🔎",
    layout="wide",
)

PUBLICATION_COL = "Publication Number"
TITLE_COL = "Title"
ABSTRACT_COL = "Abstract"
DESCRIPTION_COL = "Description"
CLAIMS_COL = "Claims"
EMBEDDING_MODEL = "text-embedding-3-small"

EMBEDDING_CHUNK_CHARS = 3_000
QUERY_MAX_CHARS = 6_000
EMBEDDING_BATCH_SIZE = 64

COLUMN_ALIASES = {
    PUBLICATION_COL: [
        "publication number", "publication_number",   "Publication_Number" ,
         "Publication Number"
    ],

    TITLE_COL: ["title", "patent title", "Title" , "invention title"],

    ABSTRACT_COL: ["abstract", "patent abstract" , "Abstract"],

    DESCRIPTION_COL: [
         "description", "Description" , "patent description", "detailed description",
        "full description", "Full Description"
        
    ],

    CLAIMS_COL: [
           
            "Claims" , "claims"
        ],
}

def normalize_header(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()

def prepare_patents_dataframe(source_df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the uploaded patent dataset and create searchable text."""
    if source_df is None or source_df.empty:
        raise ValueError("The uploaded patent dataset is empty.")

    df = source_df.copy()

    lookup = {normalize_header(column): column for column in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        source_column = next(
            (column for column in df.columns
             if normalize_header(column) == normalize_header(canonical)),
            None,
        )
        if source_column is None:
            for alias in aliases:
                matched = lookup.get(normalize_header(alias))
                if matched is not None:
                    source_column = matched
                    break

        if source_column is None:
            # Abstract and Description are display/context fields only.
            
            if canonical in {ABSTRACT_COL, DESCRIPTION_COL}:
                df[canonical] = ""
            else:
                raise ValueError(
                    f"Required column '{canonical}' was not found. "
                    f"Accepted alternatives: {', '.join(aliases)}"
                )
        elif source_column != canonical:
            df[canonical] = df[source_column]

    for column in [
        PUBLICATION_COL, TITLE_COL, ABSTRACT_COL, DESCRIPTION_COL, CLAIMS_COL
    ]:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].fillna("").astype(str).str.strip()

    df["search_text"] = (
         
            df[ABSTRACT_COL]
                    .str.replace(r"\s+", " ", regex=True)
                    .str.strip()
                    +
                    " "
                    +
             df[CLAIMS_COL]
                            .str.replace(r"\s+", " ", regex=True)
                            .str.strip() 
    )

    df = df[df["search_text"].str.len() > 0].copy().reset_index(drop=True)

    if df.empty:
        raise ValueError("The dataset contains no searchable patent records.")

    missing_ids = df[PUBLICATION_COL].eq("")

    if missing_ids.any():
        missing_count = int(missing_ids.sum())
        
        df.loc[missing_ids, PUBLICATION_COL] = [
            f"Unknown publication #{index + 1}" for index in range(missing_count)
        ]
    return df

def get_ranks(scores: list[float] | np.ndarray) -> list[int]:
    """Convert scores to descending, one-based ranks."""
    values = np.asarray(scores, dtype=float)
    ordering = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=int)
    ranks[ordering] = np.arange(1, len(values) + 1)
    return ranks.tolist()

def tfidf_search(query: str, documents: list[str]) -> list[float]:
    corpus = [query] + documents
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=300_000,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)
    return cosine_similarity(matrix[0], matrix[1:]).ravel().tolist()

def get_openai_api_key() -> str | None:
    """Read the key from Streamlit secrets first, then environment variables."""
    try:
        key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None
    return key or os.getenv("OPENAI_API_KEY")


def split_for_embeddings(text: str, max_chars: int = EMBEDDING_CHUNK_CHARS) -> list[str]:
    """Split long patent text into overlapping, API-safe embedding chunks."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]

    # Small overlap helps avoid splitting a technical phrase at a boundary.
    overlap = 200
    step = max_chars - overlap
    chunks = [text[start:start + max_chars] for start in range(0, len(text), step)]
    return [chunk for chunk in chunks if chunk.strip()]

@st.cache_data(show_spinner=False, ttl=86400)
def get_document_embeddings(
    documents: tuple[str, ...], model_name: str, api_key: str
) -> tuple[np.ndarray, np.ndarray]:
    """Embed patent chunks once; return chunk vectors and their patent indexes."""
    if OpenAI is None:
        raise RuntimeError("OpenAI package is missing")

    chunks: list[str] = []
    patent_indexes: list[int] = []
    for patent_index, document in enumerate(documents):
        for chunk in split_for_embeddings(document):
            chunks.append(chunk)
            patent_indexes.append(patent_index)

    client = OpenAI(api_key=api_key)
    vectors: list[list[float] | None] = [None] * len(chunks)

    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start:start + EMBEDDING_BATCH_SIZE]
        response = client.embeddings.create(model=model_name, input=batch)
        for item in response.data:
            vectors[start + item.index] = item.embedding

    if any(vector is None for vector in vectors):
        raise RuntimeError("The embedding API did not return an embedding for every patent chunk.")
    return np.asarray(vectors, dtype=np.float32), np.asarray(patent_indexes, dtype=np.int32)

def semantic_search(
    query: str,
    documents: list[str],
    api_key: str,
    model_name: str = EMBEDDING_MODEL,
) -> list[float]:
    """Score each patent by its most semantically similar text chunk."""
    if OpenAI is None:
        raise RuntimeError("OpenAI package is missing")

    client = OpenAI(api_key=api_key)
    safe_query = str(query).strip()[:QUERY_MAX_CHARS]
    query_response = client.embeddings.create(model=model_name, input=safe_query)
    query_vector = np.asarray(query_response.data[0].embedding, dtype=np.float32)

    chunk_vectors, patent_indexes = get_document_embeddings(
        tuple(documents), model_name, api_key
    )

    # Cosine similarity between the query and every chunk.
    query_norm = np.linalg.norm(query_vector)
    chunk_norms = np.linalg.norm(chunk_vectors, axis=1)
    denominator = chunk_norms * query_norm
    denominator[denominator == 0] = 1.0
    chunk_scores = (chunk_vectors @ query_vector) / denominator

    # material found deep in a long description while returning one score per patent.
    patent_scores = np.full(len(documents), -1.0, dtype=np.float32)
    for chunk_index, patent_index in enumerate(patent_indexes):
        patent_scores[patent_index] = max(
            patent_scores[patent_index], float(chunk_scores[chunk_index])
        )
    return patent_scores.astype(float).tolist()

def retrieve_patents(
    source_df: pd.DataFrame,
    product_description: str,
    similarity_threshold: float = 0.02,
    semantic_threshold: float = 0.33,
    use_openai_embeddings: bool = True,
    rrf_k: int = 60,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Retrieve candidates if they qualify via TF-IDF or semantic score."""
    query = str(product_description or "").strip()
    if not query:
        raise ValueError("Please enter a product description.")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("TF-IDF threshold must be between 0 and 1.")
    if not 0.0 <= semantic_threshold <= 1.0:
        raise ValueError("Semantic threshold must be between 0 and 1.")

    df = prepare_patents_dataframe(source_df)
    documents = df["search_text"].tolist()

    tfidf_scores = tfidf_search(query, documents)
    df["TF-IDF Similarity"] = tfidf_scores
    rankings: dict[str, list[float]] = {"TF-IDF": tfidf_scores}

    semantic_scores: list[float] | None = None
    if use_openai_embeddings:
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError(
                "OpenAI API key not found"
                
            )
        
        semantic_scores = semantic_search(query, documents, api_key)
        df["Semantic Similarity"] = semantic_scores
        rankings["Semantic"] = semantic_scores

    # Combined retrieval rule: a patent qualifies through ANY enabled method.
    qualifying_mask = df["TF-IDF Similarity"] >= similarity_threshold
    if semantic_scores is not None:
        qualifying_mask |= df["Semantic Similarity"] >= semantic_threshold

    df["Search Score"] = 0.0
    for method_name, scores in rankings.items():
        ranks = get_ranks(scores)
        df[f"{method_name} Rank"] = ranks
        df["Search Score"] += [1.0 / (rrf_k + rank) for rank in ranks]

    candidates = df.loc[qualifying_mask].copy()
    candidates = candidates.sort_values(
        ["Search Score", "TF-IDF Similarity"], ascending=False
    ).reset_index(drop=True)
    candidates["Search Rank"] = range(1, len(candidates) + 1)

    metadata = {
        "methods_used": list(rankings.keys()),
        "threshold_method": "TF-IDF OR semantic similarity",
        "similarity_threshold": similarity_threshold,
        "semantic_threshold": semantic_threshold if semantic_scores is not None else None,
        "semantic_search_used": semantic_scores is not None,
        "embedding_model": EMBEDDING_MODEL if semantic_scores is not None else None,
        "cross_encoder_used": False,
        "corpus_size": len(df),
        "candidates_returned": len(candidates),
    }
    return candidates, metadata


def main() -> None:
    st.title("Patent Search")
    st.caption(
        "Search the product description. This retrieves the relevant patent candidates"
    )
    st.divider()

    with st.form("patent_search_form"):
        product_description = st.text_area(
            "Product Description",
            height=220,
            placeholder=(
                "Describe the product, its technical operation, components, "
                "and intended functionality..."
            ),
        )
        uploaded_file = st.file_uploader(
            "Upload Patent Corpus (.xlsx)",
            type=["xlsx"],
            help=(
                "Required columns: Publication Number, Title, Abstract, Description and Claims. "
                
            ),
        )
            
            # similarity_threshold = st.slider(
            #     "Minimum TF-IDF similarity threshold",
            #     min_value=0.0, max_value=1.0, value=0.10, step=0.01,
            #     help="A patent qualifies through TF-IDF if its cosine similarity meets this threshold.",
            # )
            # semantic_threshold = st.slider(
            #     "Minimum semantic similarity threshold",
            #     min_value=0.0, max_value=1.0, value=0.30, step=0.01,
            #     help=(
            #         "A patent also qualifies if its OpenAI embedding cosine similarity meets this threshold. "
                    
            #     ),
            # )

        similarity_threshold = 0.02
        semantic_threshold = 0.33
            

        # with col2:
        #     st.text_input(
        #         "Embedding model", value=EMBEDDING_MODEL, disabled=True,
        #         help="Hosted OpenAI model",
        #     )

        # use_openai_embeddings = st.checkbox(
        #     "Use OpenAI semantic embeddings (requires API key and API billing)",
        #     value=True,
        # )

        submitted = st.form_submit_button(
            "Search Patents", type="primary", use_container_width=True
        )

    if submitted:
        if not product_description.strip():
            st.error("Please enter a product description.")
        elif uploaded_file is None:
            st.error("Please upload the patent corpus Excel file.")
        else:
            try:
                source_df = pd.read_excel(uploaded_file, engine="openpyxl")
                with st.spinner(
                    "Searching patents... OpenAI may embed the corpus on the first run; cached embeddings are reused afterward."
                ):
                    candidates, metadata = retrieve_patents(
                        source_df=source_df,
                        product_description=product_description,
                        similarity_threshold=similarity_threshold,
                        semantic_threshold=semantic_threshold,
                        # use_openai_embeddings=use_openai_embeddings,
                    )
                st.session_state["patent_search_candidates"] = candidates
                st.session_state["patent_search_metadata"] = metadata
                st.session_state["patent_search_product_description"] = product_description
                st.session_state["patent_search_corpus_name"] = uploaded_file.name
                st.success(
                    f"Retrieved {len(candidates)} qualifying patents from "
                    f"{metadata['corpus_size']} searchable records."
                )
            except Exception as exc:
                st.error(f"Patent search failed: {exc}")

    candidates = st.session_state.get("patent_search_candidates")
    metadata = st.session_state.get("patent_search_metadata")
    if candidates is not None and metadata is not None:
        st.divider()
        st.subheader("Retrieved Patent Candidates")
        semantic_status = (
            f"OpenAI embeddings ({metadata['embedding_model']})"
            if metadata["semantic_search_used"] else "OpenAI embeddings disabled"
        )
        st.caption(
            f"Corpus: {st.session_state.get('patent_search_corpus_name', 'Uploaded file')} | "
            f"Records searched: {metadata['corpus_size']} | "
            f"Inclusion rule: {metadata['threshold_method']} | "
            f"TF-IDF threshold: {metadata['similarity_threshold']:.2f} | "
            f"Semantic: {semantic_status} | "
            f"Ranking methods: {', '.join(metadata['methods_used'])}"
        )
        st.metric("Qualifying patents", len(candidates))

        if candidates.empty:
            st.warning(
                "No patents met the selected retrieval rule"
            )
        else:
            preferred_columns = [
                PUBLICATION_COL, TITLE_COL, ABSTRACT_COL,
                "TF-IDF Similarity", "TF-IDF Rank",
                "Semantic Similarity", "Semantic Rank", "Search Rank", "Search Score",
            ]
            display_columns = [
                column for column in preferred_columns if column in candidates.columns
            ]
            st.dataframe(
                candidates[display_columns],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Search Rank": st.column_config.NumberColumn("Search Rank", format="%d"),
                    "Search Score": st.column_config.NumberColumn("RRF Search Score", format="%.6f"),
                    "TF-IDF Similarity": st.column_config.NumberColumn("TF-IDF Similarity", format="%.4f"),
                    "Semantic Similarity": st.column_config.NumberColumn("Semantic Similarity", format="%.4f"),
                },
            )
            csv_data = candidates[display_columns].to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "Download All Retrieved Patents (CSV)",
                data=csv_data,
                file_name="RelevanceIQ_Patent_Search_Results.csv",
                mime="text/csv",
            )

            with st.expander("Inspect Patent Details"):
                for _, row in candidates.iterrows():
                    publication_number = row.get(PUBLICATION_COL, "Unknown publication")
                    title = row.get(TITLE_COL, "Untitled patent")
                    with st.expander(f"{publication_number} — {title}"):
                        st.markdown("**Abstract**")
                        st.write(row.get(ABSTRACT_COL, "") or "No abstract provided.")
                        # st.markdown("**Description**")
                        # st.write(row.get(DESCRIPTION_COL, "") or "No description provided.")
                        # st.markdown("**Claims**")
                        # st.write(row.get(CLAIMS_COL, "") or "No Claims provided.")

        st.divider()
        st.subheader("Next Step")
        st.info(
            "Patent search is complete "
           
        )

if __name__ == "__main__":
    main()




# from __future__ import annotations

# import hashlib
# import json
# import os
# import re
# import time
# import zipfile
# from collections import deque
# from concurrent.futures import ThreadPoolExecutor
# from pathlib import Path
# from typing import Any

# import numpy as np
# import pandas as pd
# import streamlit as st
# from openai import OpenAI
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# # =============================================================================
# # RelevanceIQ — Patent Retrieval + Validation
# # =============================================================================

# st.set_page_config(
#     page_title="RelevanceIQ | Patent Search",
#     page_icon="🔎",
#     layout="wide",
# )

# PUBLICATION_COL = "Publication Number"
# TITLE_COL = "Title"
# ABSTRACT_COL = "Abstract"
# DESCRIPTION_COL = "Description"
# CLAIMS_COL = "Claims"

# EMBEDDING_MODEL = "text-embedding-3-small"
# EMBEDDING_CHUNK_CHARS = 3000
# QUERY_MAX_CHARS = 6000
# EMBEDDING_BATCH_SIZE = 64
# EMBEDDING_WORKERS = 6
# EMBEDDING_CACHE_TTL_SECONDS = 24 * 60 * 60
# MIN_RESULTS_FOR_RELATIVE_CHECKS = 10

# BASE_DIR = Path("/content") if Path("/content").exists() else Path.cwd()
# CACHE_DIR = Path(
#     os.getenv("RELEVANCEIQ_CACHE_DIR", str(BASE_DIR / "relevanceiq_cache"))
# )

# COLUMN_ALIASES = {
#     PUBLICATION_COL: ["publication number", "publication_number"],
#     TITLE_COL: ["title", "patent title", "invention title"],
#     ABSTRACT_COL: ["abstract", "patent abstract"],
#     DESCRIPTION_COL: [
#         "description", "patent description", "detailed description",
#         "full description",
#     ],
#     CLAIMS_COL: ["claims"],
# }

# _CORPUS_CACHE: dict = {}
# _EMBED_CACHE: dict = {}


# # =============================================================================
# # DATA PREPARATION
# # =============================================================================

# def normalize_header(value: Any) -> str:
#     return re.sub(r"\s+", " ", str(value)).strip().lower()


# def prepare_patents_dataframe(source_df: pd.DataFrame) -> pd.DataFrame:
#     """The initial retrieval text remains Abstract + Claims."""
#     if source_df is None or source_df.empty:
#         raise ValueError("The uploaded patent dataset is empty.")

#     df = source_df.copy()
#     lookup = {normalize_header(c): c for c in df.columns}

#     for canonical, aliases in COLUMN_ALIASES.items():
#         source_column = lookup.get(normalize_header(canonical))

#         if source_column is None:
#             for alias in aliases:
#                 if normalize_header(alias) in lookup:
#                     source_column = lookup[normalize_header(alias)]
#                     break

#         if source_column is None:
#             if canonical in {ABSTRACT_COL, DESCRIPTION_COL}:
#                 df[canonical] = ""
#             else:
#                 raise ValueError(
#                     f"Required column '{canonical}' was not found. "
#                     f"Accepted alternatives: {', '.join(aliases)}"
#                 )
#         elif source_column != canonical:
#             df[canonical] = df[source_column]

#     for column in [
#         PUBLICATION_COL, TITLE_COL, ABSTRACT_COL, DESCRIPTION_COL, CLAIMS_COL
#     ]:
#         if column not in df.columns:
#             df[column] = ""
#         df[column] = df[column].fillna("").astype(str).str.strip()

#     # EXACT original retrieval text: Abstract + Claims.
#     # Title and Description are not included in the retrieval text.
#     df["search_text"] = (
#         df[ABSTRACT_COL].str.replace(r"\s+", " ", regex=True).str.strip()
#         + " "
#         + df[CLAIMS_COL].str.replace(r"\s+", " ", regex=True).str.strip()
#     ).str.strip()

#     df = df[df["search_text"].str.len() > 0].copy().reset_index(drop=True)

#     if df.empty:
#         raise ValueError("The dataset contains no searchable patent records.")

#     missing_ids = df[PUBLICATION_COL].eq("")
#     if missing_ids.any():
#         count = int(missing_ids.sum())
#         df.loc[missing_ids, PUBLICATION_COL] = [
#             f"Unknown publication #{i + 1}" for i in range(count)
#         ]

#     return df


# def _file_sha1(path: str) -> str:
#     digest = hashlib.sha1()
#     with open(path, "rb") as handle:
#         for block in iter(lambda: handle.read(8 << 20), b""):
#             digest.update(block)
#     return digest.hexdigest()


# def _read_xlsx_streaming(path: str) -> pd.DataFrame:
#     """Read large XLSX files in read-only mode."""
#     from openpyxl import load_workbook

#     workbook = load_workbook(path, read_only=True, data_only=True)
#     try:
#         rows = workbook.active.iter_rows(values_only=True)
#         header = next(rows, None)

#         if header is None:
#             raise ValueError("The first sheet of the Excel file is empty.")

#         names = ["" if value is None else str(value) for value in header]
#         columns = [[] for _ in names]

#         for row in rows:
#             for index, column in enumerate(columns):
#                 column.append(row[index] if index < len(row) else None)
#     finally:
#         workbook.close()

#     frame = pd.DataFrame(dict(enumerate(columns)))
#     frame.columns = names
#     return frame


# def read_corpus_table(path: str) -> pd.DataFrame:
#     size = os.path.getsize(path)
#     if size == 0:
#         raise ValueError("The uploaded corpus file is empty.")

#     with open(path, "rb") as handle:
#         head = handle.read(1024)

#     if head.startswith(b"PK\x03\x04"):
#         if not zipfile.is_zipfile(path):
#             raise ValueError("The Excel file appears incomplete or corrupted.")
#         return _read_xlsx_streaming(path)

#     if head.startswith(b"\xd0\xcf\x11\xe0"):
#         try:
#             return pd.read_excel(path, engine="xlrd")
#         except ImportError as exc:
#             raise ValueError(
#                 "This is an old XLS file. Save it as XLSX or install xlrd."
#             ) from exc

#     if head.lstrip()[:1] == b"<":
#         raise ValueError("The uploaded file appears to be HTML, not Excel or CSV.")

#     last_error = None
#     for encoding in ("utf-8-sig", "latin-1"):
#         try:
#             return pd.read_csv(path, encoding=encoding)
#         except Exception as exc:
#             last_error = exc

#     raise ValueError(f"Could not read the corpus as Excel or CSV: {last_error}")


# def load_prepared_corpus(path: str) -> tuple[pd.DataFrame, str]:
#     file_key = _file_sha1(path)

#     if file_key in _CORPUS_CACHE:
#         return _CORPUS_CACHE[file_key]

#     df = prepare_patents_dataframe(read_corpus_table(path))
#     digest = hashlib.sha1()

#     for text in df["search_text"]:
#         digest.update(text.encode("utf-8", "ignore"))
#         digest.update(b"\0")

#     corpus_key = digest.hexdigest()
#     result = (df, corpus_key)
#     _CORPUS_CACHE[file_key] = result

#     while len(_CORPUS_CACHE) > 2:
#         _CORPUS_CACHE.pop(next(iter(_CORPUS_CACHE)))

#     return result


# # =============================================================================
# # ORIGINAL TF-IDF METHOD — UNCHANGED
# # =============================================================================

# def get_ranks(scores) -> list[int]:
#     values = np.asarray(scores, dtype=float)
#     ordering = np.argsort(-values, kind="stable")
#     ranks = np.empty(len(values), dtype=int)
#     ranks[ordering] = np.arange(1, len(values) + 1)
#     return ranks.tolist()


# def tfidf_search(query: str, documents: list[str]) -> list[float]:
#     """The original TF-IDF settings and cosine-similarity calculation."""
#     corpus = [query] + documents

#     vectorizer = TfidfVectorizer(
#         lowercase=True,
#         stop_words="english",
#         ngram_range=(1, 2),
#         max_features=300_000,
#         sublinear_tf=True,
#     )

#     matrix = vectorizer.fit_transform(corpus)
#     return cosine_similarity(matrix[0], matrix[1:]).ravel().tolist()

# # =============================================================================
# # OPENAI EMBEDDINGS — ORIGINAL SCORING + 24-HOUR DISK CACHE
# # =============================================================================

# def get_openai_api_key() -> str:
#     key = os.getenv("OPENAI_API_KEY")

#     if not key:
#         try:
#             key = st.secrets.get("OPENAI_API_KEY")
#         except Exception:
#             key = None

#     if not key:
#         raise ValueError(
#             "OPENAI_API_KEY is not set. Add it to your environment or "
#             "Streamlit secrets."
#         )

#     return key


# def split_for_embeddings(
#     text: str,
#     max_chars: int = EMBEDDING_CHUNK_CHARS,
# ) -> list[str]:
#     text = re.sub(r"\s+", " ", str(text or "")).strip()

#     if not text:
#         return [""]
#     if len(text) <= max_chars:
#         return [text]

#     overlap = 200
#     step = max_chars - overlap

#     return [
#         text[start:start + max_chars]
#         for start in range(0, len(text), step)
#         if text[start:start + max_chars].strip()
#     ]


# def _embed_batch(
#     client: OpenAI,
#     batch: list[str],
#     model_name: str,
#     retries: int = 6,
# ) -> list[list[float]]:
#     for attempt in range(retries):
#         try:
#             response = client.embeddings.create(model=model_name, input=batch)
#             return [
#                 item.embedding
#                 for item in sorted(response.data, key=lambda item: item.index)
#             ]
#         except Exception:
#             if attempt == retries - 1:
#                 raise
#             time.sleep(min(2 ** attempt, 30))


# def get_document_embeddings(
#     corpus_key: str,
#     model_name: str,
#     documents: list[str],
#     api_key: str,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Resumable disk cache. Cache files expire after 24 hours.

#     This function deliberately does not call Streamlit UI/progress functions.
#     """
#     memo_key = f"{corpus_key}|{model_name}"

#     if memo_key in _EMBED_CACHE:
#         return _EMBED_CACHE[memo_key]

#     CACHE_DIR.mkdir(parents=True, exist_ok=True)

#     file_key = hashlib.sha1(
#         f"{corpus_key}|{model_name}|{EMBEDDING_CHUNK_CHARS}".encode()
#     ).hexdigest()[:24]

#     vec_path = CACHE_DIR / f"{file_key}_vectors.npy"
#     idx_path = CACHE_DIR / f"{file_key}_indexes.npy"
#     meta_path = CACHE_DIR / f"{file_key}_meta.json"

#     meta = {}
#     if meta_path.exists():
#         try:
#             meta = json.loads(meta_path.read_text())
#         except Exception:
#             meta = {}

#     timestamp = meta.get("updated_at", meta.get("created_at", 0))
#     expired = (
#         not timestamp
#         or time.time() - float(timestamp) > EMBEDDING_CACHE_TTL_SECONDS
#     )

#     if expired:
#         for path in (vec_path, idx_path, meta_path):
#             try:
#                 path.unlink(missing_ok=True)
#             except Exception:
#                 pass
#         meta = {}

#     if (
#         meta.get("complete")
#         and vec_path.exists()
#         and idx_path.exists()
#     ):
#         result = (
#             np.load(vec_path, mmap_mode="r"),
#             np.load(idx_path),
#         )
#         _EMBED_CACHE[memo_key] = result
#         return result

#     chunks = []
#     owners = []

#     for patent_index, document in enumerate(documents):
#         for chunk in split_for_embeddings(document):
#             chunks.append(chunk)
#             owners.append(patent_index)

#     indexes = np.asarray(owners, dtype=np.int32)
#     total = len(chunks)

#     if not total:
#         raise ValueError("No patent text was available for semantic embedding.")

#     client = OpenAI(api_key=api_key, max_retries=5)
#     vectors = None
#     done = 0

#     # Resume a compatible incomplete cache.
#     if (
#         meta.get("n_chunks") == total
#         and meta.get("dim")
#         and vec_path.exists()
#     ):
#         try:
#             vectors = np.lib.format.open_memmap(vec_path, mode="r+")
#             if vectors.shape == (total, int(meta["dim"])):
#                 done = max(0, min(int(meta.get("done", 0)), total))
#             else:
#                 vectors = None
#         except Exception:
#             vectors = None

#     if vectors is None:
#         first_batch = chunks[:EMBEDDING_BATCH_SIZE]
#         first_vectors = _embed_batch(client, first_batch, model_name)

#         vectors = np.lib.format.open_memmap(
#             vec_path,
#             mode="w+",
#             dtype=np.float32,
#             shape=(total, len(first_vectors[0])),
#         )

#         vectors[:len(first_vectors)] = np.asarray(first_vectors, dtype=np.float32)
#         done = len(first_vectors)

#     dim = int(vectors.shape[1])
#     next_start = done
#     last_save = time.time()
#     workers = max(1, EMBEDDING_WORKERS)
#     pending = deque()

#     try:
#         with ThreadPoolExecutor(max_workers=workers) as pool:
#             while next_start < total or pending:
#                 while next_start < total and len(pending) < workers * 3:
#                     batch = chunks[next_start:next_start + EMBEDDING_BATCH_SIZE]
#                     pending.append((
#                         next_start,
#                         pool.submit(_embed_batch, client, batch, model_name),
#                     ))
#                     next_start += len(batch)

#                 start, future = pending.popleft()
#                 result = future.result()

#                 vectors[start:start + len(result)] = np.asarray(
#                     result, dtype=np.float32
#                 )
#                 done = start + len(result)

#                 if time.time() - last_save > 20:
#                     vectors.flush()
#                     meta_path.write_text(json.dumps({
#                         "n_chunks": total,
#                         "dim": dim,
#                         "done": done,
#                         "complete": False,
#                         "updated_at": time.time(),
#                     }))
#                     last_save = time.time()

#     except BaseException:
#         for _, future in pending:
#             future.cancel()

#         vectors.flush()
#         meta_path.write_text(json.dumps({
#             "n_chunks": total,
#             "dim": dim,
#             "done": done,
#             "complete": False,
#             "updated_at": time.time(),
#         }))
#         raise

#     vectors.flush()
#     np.save(idx_path, indexes)

#     now = time.time()
#     meta_path.write_text(json.dumps({
#         "n_chunks": total,
#         "dim": dim,
#         "done": total,
#         "complete": True,
#         "created_at": now,
#         "updated_at": now,
#     }))

#     del vectors

#     result = (np.load(vec_path, mmap_mode="r"), indexes)
#     _EMBED_CACHE[memo_key] = result

#     while len(_EMBED_CACHE) > 2:
#         _EMBED_CACHE.pop(next(iter(_EMBED_CACHE)))

#     return result


# def semantic_search(
#     query: str,
#     documents: list[str],
#     api_key: str,
#     corpus_key: str,
#     model_name: str = EMBEDDING_MODEL,
# ) -> list[float]:
#     """Original semantic method: highest chunk cosine similarity per patent."""
#     client = OpenAI(api_key=api_key, max_retries=5)
#     safe_query = str(query).strip()[:QUERY_MAX_CHARS]

#     response = client.embeddings.create(
#         model=model_name,
#         input=safe_query,
#     )
#     query_vector = np.asarray(response.data[0].embedding, dtype=np.float32)

#     chunk_vectors, patent_indexes = get_document_embeddings(
#         corpus_key, model_name, documents, api_key
#     )

#     query_norm = float(np.linalg.norm(query_vector))
#     chunk_scores = np.empty(len(chunk_vectors), dtype=np.float32)

#     for start in range(0, len(chunk_vectors), 50_000):
#         end = min(start + 50_000, len(chunk_vectors))
#         block = np.asarray(chunk_vectors[start:end])
#         norms = np.linalg.norm(block, axis=1)
#         denominator = norms * query_norm
#         denominator[denominator == 0] = 1.0
#         chunk_scores[start:end] = (block @ query_vector) / denominator

#     starts = np.flatnonzero(np.r_[True, np.diff(patent_indexes) != 0])
#     patent_scores = np.full(len(documents), -1.0, dtype=np.float32)
#     patent_scores[patent_indexes[starts]] = np.maximum.reduceat(chunk_scores, starts)

#     return patent_scores.astype(float).tolist()

# # =============================================================================
# # VALIDATION SIGNALS
# # =============================================================================

# def robust_z(values) -> np.ndarray:
#     values = np.asarray(values, dtype=float)
#     median = np.median(values)
#     mad = np.median(np.abs(values - median)) * 1.4826

#     if mad < 1e-9:
#         return np.zeros_like(values)

#     return (values - median) / mad


# def validation_key_terms(
#     query: str,
#     texts: list[str],
#     claims: list[str],
#     top_n: int = 30,
# ) -> dict:
#     n = len(texts)

#     empty = {
#         "terms": [],
#         "term_coverage": np.zeros(n),
#         "claims_term_coverage": np.zeros(n),
#         "terms_matched": np.zeros(n, dtype=int),
#         "matched_terms": [""] * n,
#     }

#     settings = {
#         "lowercase": True,
#         "stop_words": "english",
#         "ngram_range": (1, 2),
#     }

#     try:
#         vocabulary = CountVectorizer(**settings).fit([query]).vocabulary_
#     except ValueError:
#         return empty

#     counter = CountVectorizer(vocabulary=vocabulary, **settings)
#     full_counts = counter.transform(texts)
#     claims_counts = counter.transform(claims)
#     query_counts = counter.transform([query]).toarray().ravel()

#     presence = full_counts > 0
#     doc_freq = np.asarray(presence.sum(axis=0)).ravel()

#     eligible = (
#         (doc_freq >= max(2, 0.02 * n))
#         & (doc_freq <= 0.95 * n)
#         & (query_counts > 0)
#     )

#     if not eligible.any():
#         return empty

#     idf = np.log((1 + n) / (1 + doc_freq)) + 1.0
#     weights_all = np.zeros(len(query_counts))
#     weights_all[eligible] = (
#         (1.0 + np.log(query_counts[eligible])) * idf[eligible]
#     )

#     top_idx = np.argsort(-weights_all)[:top_n]
#     top_idx = top_idx[weights_all[top_idx] > 0]

#     if len(top_idx) == 0:
#         return empty

#     weights = np.sqrt(weights_all[top_idx])
#     index_to_term = np.empty(len(vocabulary), dtype=object)

#     for term, index in vocabulary.items():
#         index_to_term[index] = term

#     terms = index_to_term[top_idx]
#     full_presence = presence[:, top_idx].astype(float).toarray()
#     claims_presence = (claims_counts[:, top_idx] > 0).astype(float).toarray()

#     return {
#         "terms": terms.tolist(),
#         "term_coverage": (full_presence @ weights) / weights.sum(),
#         "claims_term_coverage": (claims_presence @ weights) / weights.sum(),
#         "terms_matched": full_presence.sum(axis=1).astype(int),
#         "matched_terms": [", ".join(terms[row > 0]) for row in full_presence],
#     }


# def split_into_features(
#     text: str,
#     min_chars: int = 40,
#     max_features: int = 15,
# ) -> list[str]:
#     parts = re.split(r"(?<=[.;:!?])\s+|\n+", str(text or ""))
#     features = [re.sub(r"\s+", " ", part).strip() for part in parts]
#     features = [item for item in features if len(item) >= min_chars]

#     if len(features) < 2:
#         flat = re.sub(r"\s+", " ", str(text or "")).strip()
#         features = [flat[i:i + 300] for i in range(0, len(flat), 300)]

#     if len(features) > max_features:
#         keep = np.linspace(0, len(features) - 1, max_features).astype(int)
#         features = [features[i] for i in keep]

#     return features


# def _unit(matrix: np.ndarray) -> np.ndarray:
#     norms = np.linalg.norm(matrix, axis=1, keepdims=True)
#     norms[norms == 0] = 1.0
#     return matrix / norms


# def retrieved_feature_scores(
#     feature_vectors,
#     chunk_vectors,
#     patent_indexes,
#     candidate_doc_idx,
# ) -> np.ndarray:
#     keep_idx = np.flatnonzero(np.isin(patent_indexes, candidate_doc_idx))
#     sub_idx = patent_indexes[keep_idx]
#     unit_features = _unit(feature_vectors)
#     similarities = np.empty(
#         (len(unit_features), len(keep_idx)),
#         dtype=np.float32,
#     )

#     for start in range(0, len(keep_idx), 20_000):
#         end = min(start + 20_000, len(keep_idx))
#         block = np.asarray(chunk_vectors[keep_idx[start:end]])
#         norms = np.linalg.norm(block, axis=1)
#         norms[norms == 0] = 1.0
#         similarities[:, start:end] = (unit_features @ block.T) / norms

#     starts = np.flatnonzero(np.r_[True, np.diff(sub_idx) != 0])
#     per_patent = np.maximum.reduceat(similarities, starts, axis=1)
#     order = np.searchsorted(sub_idx[starts], candidate_doc_idx)

#     return per_patent[:, order]


# def feature_coverage(scores: np.ndarray, percentile: float = 50.0):
#     cutoffs = np.percentile(scores, percentile, axis=1, keepdims=True)
#     covered = scores >= cutoffs
#     return covered.sum(axis=0), covered.mean(axis=0)


# def add_validation_signals(
#     candidates: pd.DataFrame,
#     query: str,
#     candidate_doc_idx,
#     client=None,
#     chunk_vectors=None,
#     patent_indexes=None,
#     model_name=EMBEDDING_MODEL,
#     feature_percentile=50.0,
#     top_n_terms=30,
# ) -> dict:
#     info = {"key_terms": [], "n_features": 0}

#     signals = validation_key_terms(
#         query,
#         candidates["search_text"].tolist(),
#         candidates[CLAIMS_COL].tolist(),
#         top_n_terms,
#     )

#     candidates["Term Coverage"] = signals["term_coverage"]
#     candidates["Claims Term Coverage"] = signals["claims_term_coverage"]
#     candidates["Key Terms Matched"] = signals["terms_matched"]
#     candidates["Matched Key Terms"] = signals["matched_terms"]
#     info["key_terms"] = signals["terms"]

#     if len(candidates) < MIN_RESULTS_FOR_RELATIVE_CHECKS:
#         info["note"] = (
#             "Too few retrieved patents for relative checks; only term checks were run."
#         )
#         return info

#     if "Semantic Similarity" in candidates.columns:
#         candidates["Semantic Z"] = robust_z(candidates["Semantic Similarity"])

#     if client is not None and chunk_vectors is not None and patent_indexes is not None:
#         features = split_into_features(query)

#         if features:
#             response = client.embeddings.create(model=model_name, input=features)
#             feature_vectors = np.asarray(
#                 [
#                     item.embedding
#                     for item in sorted(response.data, key=lambda item: item.index)
#                 ],
#                 dtype=np.float32,
#             )

#             scores = retrieved_feature_scores(
#                 feature_vectors,
#                 chunk_vectors,
#                 patent_indexes,
#                 candidate_doc_idx,
#             )

#             counts, shares = feature_coverage(scores, feature_percentile)
#             candidates["Feature Coverage"] = shares
#             candidates["Features Covered"] = [
#                 f"{count}/{len(features)}" for count in counts
#             ]
#             info["n_features"] = len(features)

#     return info


# def apply_validation(
#     candidates: pd.DataFrame,
#     n_key_terms: int,
#     min_term_coverage=0.10,
#     min_claim_term_coverage=0.05,
#     min_terms_matched=2,
#     min_feature_coverage=0.20,
#     semantic_z_cutoff=2.0,
#     min_pass_share=0.40,
# ) -> pd.DataFrame:
#     """Validation filter. Does not recalculate retrieval scores or ranks."""
#     out = candidates.copy()
#     required = min(int(min_terms_matched), max(n_key_terms, 0))

#     checks = {
#         "Key terms": out["Term Coverage"].to_numpy() >= min_term_coverage,
#         "Claim terms": out["Claims Term Coverage"].to_numpy() >= min_claim_term_coverage,
#         "Terms matched": out["Key Terms Matched"].to_numpy() >= required,
#     }

#     if "Feature Coverage" in out.columns:
#         checks["Features"] = out["Feature Coverage"].to_numpy() >= min_feature_coverage

#     if "Semantic Z" in out.columns:
#         checks["Semantic"] = out["Semantic Z"].to_numpy() >= -semantic_z_cutoff

#     total = len(checks)
#     passed = np.sum(list(checks.values()), axis=0)
#     no_key_terms = out["Key Terms Matched"].to_numpy() == 0
#     keep = (~no_key_terms) & (passed / total >= min_pass_share - 1e-9)

#     out["Validation Status"] = np.where(keep, "Kept", "Rejected")

#     for name, flags in checks.items():
#         out[f"Pass: {name}"] = flags

#     out["Checks Passed"] = [f"{count}/{total}" for count in passed]
#     return out


# # =============================================================================
# # ORIGINAL RETRIEVAL + VALIDATION AFTER RETRIEVAL
# # =============================================================================

# def retrieve_patents(
#     prepared_df: pd.DataFrame,
#     corpus_key: str,
#     product_description: str,
#     similarity_threshold: float = 0.10,
#     semantic_threshold: float = 0.30,
#     use_openai_embeddings: bool = True,
#     rrf_k: int = 60,
#     validate: bool = True,
#     feature_percentile: float = 50.0,
# ):
#     query = str(product_description or "").strip()

#     if not query:
#         raise ValueError("Please enter a product description.")
#     if not 0 <= similarity_threshold <= 1:
#         raise ValueError("TF-IDF threshold must be between 0 and 1.")
#     if not 0 <= semantic_threshold <= 1:
#         raise ValueError("Semantic threshold must be between 0 and 1.")

#     timings = {}
#     df = prepared_df.copy()
#     documents = df["search_text"].tolist()

#     t0 = time.perf_counter()
#     tfidf_scores = tfidf_search(query, documents)
#     timings["TF-IDF search"] = time.perf_counter() - t0

#     df["TF-IDF Similarity"] = tfidf_scores
#     rankings = {"TF-IDF": tfidf_scores}
#     api_key = None
#     semantic_scores = None

#     if use_openai_embeddings:
#         api_key = get_openai_api_key()
#         t0 = time.perf_counter()
#         semantic_scores = semantic_search(query, documents, api_key, corpus_key)
#         timings["Semantic search"] = time.perf_counter() - t0
#         df["Semantic Similarity"] = semantic_scores
#         rankings["Semantic"] = semantic_scores

#     # Original OR-based threshold rule.
#     mask = df["TF-IDF Similarity"] >= similarity_threshold
#     if semantic_scores is not None:
#         mask |= df["Semantic Similarity"] >= semantic_threshold

#     # Original Reciprocal Rank Fusion.
#     df["Search Score"] = 0.0
#     for method_name, scores in rankings.items():
#         ranks = get_ranks(scores)
#         df[f"{method_name} Rank"] = ranks
#         df["Search Score"] += [1.0 / (rrf_k + rank) for rank in ranks]

#     # Freeze original candidates and ranking before validation.
#     candidates = df.loc[mask].copy()
#     candidates["_doc_index"] = candidates.index
#     candidates = candidates.sort_values(
#         ["Search Score", "TF-IDF Similarity"],
#         ascending=False,
#     ).reset_index(drop=True)
#     candidates["Search Rank"] = range(1, len(candidates) + 1)

#     info = {}
#     if validate and not candidates.empty:
#         t0 = time.perf_counter()
#         client = None
#         chunk_vectors = None
#         patent_indexes = None

#         if semantic_scores is not None:
#             client = OpenAI(api_key=api_key, max_retries=5)
#             chunk_vectors, patent_indexes = get_document_embeddings(
#                 corpus_key, EMBEDDING_MODEL, documents, api_key
#             )

#         info = add_validation_signals(
#             candidates,
#             query,
#             candidates["_doc_index"].to_numpy(),
#             client,
#             chunk_vectors,
#             patent_indexes,
#             EMBEDDING_MODEL,
#             float(feature_percentile),
#         )
#         timings["Validation signals"] = time.perf_counter() - t0

#     metadata = {
#         "corpus_size": len(df),
#         "initial_candidate_count": len(candidates),
#         "validation_used": validate,
#         "validation_key_terms": info.get("key_terms", []),
#         "validation_note": info.get("note"),
#         "timings": timings,
#         "similarity_threshold": similarity_threshold,
#         "semantic_threshold": semantic_threshold if semantic_scores is not None else None,
#         "semantic_search_used": semantic_scores is not None,
#         "embedding_model": EMBEDDING_MODEL if semantic_scores is not None else None,
#         "methods_used": list(rankings.keys()),
#     }

#     return candidates, metadata

# # =============================================================================
# # FINAL VALIDATION
# # =============================================================================

# def _read_known_ids(path: str) -> set[str]:
#     known = (
#         pd.read_csv(path)
#         if path.lower().endswith(".csv")
#         else pd.read_excel(path, engine="openpyxl")
#     )

#     col = next(
#         (
#             column for column in known.columns
#             if normalize_header(column) in {"publication number", "publication_number"}
#         ),
#         known.columns[0],
#     )

#     return set(known[col].astype(str).str.strip().str.upper())


# def finalize(
#     raw: pd.DataFrame,
#     metadata: dict,
#     strict: dict,
#     known_path: str | None = None,
# ):
#     diag = ""
#     ready = metadata.get("validation_used") and "Term Coverage" in raw.columns

#     if ready and not raw.empty:
#         labelled = apply_validation(
#             raw,
#             n_key_terms=len(metadata.get("validation_key_terms", [])),
#             **strict,
#         )

#         pass_cols = [c for c in labelled.columns if c.startswith("Pass: ")]
#         rows = "\n".join(
#             f"| {column[6:]} | {int((~labelled[column]).sum())} | {(~labelled[column]).mean():.1%} |"
#             for column in pass_cols
#         )

#         diag = (
#             "**Retrieved patents failing each check**\n\n"
#             "| Check | Patents failing | Share |\n|---|---:|---:|\n"
#             + rows
#         )

#         if known_path:
#             try:
#                 known_ids = _read_known_ids(known_path)
#                 ids = labelled[PUBLICATION_COL].astype(str).str.strip().str.upper()
#                 found = ids.isin(known_ids)
#                 dropped = found & (labelled["Validation Status"] == "Rejected")

#                 diag += (
#                     f"\n\n**Known relevant patents:** {len(known_ids)}"
#                     f" | found by search: {int(found.sum())}"
#                     f" | kept: {int((found & ~dropped).sum())}"
#                     f" | dropped by validation: {int(dropped.sum())}"
#                 )
#             except Exception as exc:
#                 diag += f"\n\nCould not read known-relevant file: {exc}"

#         final = labelled[labelled["Validation Status"] != "Rejected"].reset_index(drop=True)
#     else:
#         final = raw.reset_index(drop=True).copy()

#     # Search Rank and Search Score remain the original retrieval values.
#     final["Rank"] = range(1, len(final) + 1)

#     lines = [
#         f"### Final patents: **{len(final)}**",
#         f"Initial retrieval: **{len(raw)}**",
#         f"Removed by validation: **{len(raw) - len(final)}**",
#     ]

#     if metadata.get("validation_note"):
#         lines.append(f"⚠️ {metadata['validation_note']}")

#     if metadata.get("timings"):
#         lines.append(
#             "Run time (s): "
#             + " | ".join(f"{k}: {v:.1f}" for k, v in metadata["timings"].items())
#         )

#     if len(raw) and not len(final):
#         lines.append(
#             "⚠️ Every retrieved patent was removed by validation. "
#             "Loosen the validation settings."
#         )

#     return final, "\n\n".join(lines), diag


# # =============================================================================
# # STREAMLIT UI
# # =============================================================================

# def _persist_upload(uploaded_file) -> str:
#     """Save an uploaded file so the streaming reader can use its path."""
#     file_bytes = uploaded_file.getvalue()
#     suffix = Path(uploaded_file.name).suffix or ".xlsx"
#     digest = hashlib.sha1(file_bytes).hexdigest()[:16]
#     upload_dir = BASE_DIR / "relevanceiq_uploads"
#     upload_dir.mkdir(parents=True, exist_ok=True)
#     path = upload_dir / f"{digest}{suffix}"

#     if not path.exists():
#         path.write_bytes(file_bytes)

#     return str(path)


# def _display_columns(frame: pd.DataFrame) -> list[str]:
#     preferred = [
#         "Rank", PUBLICATION_COL, TITLE_COL, ABSTRACT_COL,
#         "TF-IDF Similarity", "TF-IDF Rank",
#         "Semantic Similarity", "Semantic Rank",
#         "Search Rank", "Search Score",
#         "Term Coverage", "Claims Term Coverage",
#         "Key Terms Matched", "Matched Key Terms",
#         "Feature Coverage", "Features Covered",
#         "Validation Status", "Checks Passed",
#     ]
#     return [column for column in preferred if column in frame.columns]


# def main():
#     st.title("RelevanceIQ — Patent Search")
#     st.caption(
#         "Original retrieval: TF-IDF + OpenAI semantic similarity. "
#         "Validation runs only after the initial candidate list is created."
#     )

#     with st.form("patent_search_form"):
#         product_description = st.text_area(
#             "Product Description",
#             height=200,
#             placeholder="Describe the product, its technical operation, components, and functionality...",
#         )

#         uploaded_file = st.file_uploader(
#             "Upload Patent Corpus",
#             type=["xlsx", "xls", "csv"],
#             help="Required columns: Publication Number, Title and Claims. Abstract and Description are optional.",
#         )

#         col1, col2 = st.columns(2)
#         with col1:
#             tfidf_thr = st.slider(
#                 "Minimum TF-IDF similarity", 0.0, 1.0, 0.10, 0.01
#             )
#             sem_thr = st.slider(
#                 "Minimum semantic similarity", 0.0, 1.0, 0.30, 0.01
#             )

#         with col2:
#             use_emb = st.checkbox("Use OpenAI semantic embeddings", value=True)
#             validate = st.checkbox("Run validation after retrieval", value=True)
#             feature_pct = st.slider(
#                 "Feature-match cutoff percentile", 10, 90, 50, 5
#             )

#         st.markdown("#### Validation settings")
#         min_pass_share = st.slider(
#             "Minimum share of validation checks passed", 0.0, 1.0, 0.40, 0.05
#         )

#         c1, c2, c3 = st.columns(3)
#         with c1:
#             min_term_cov = st.slider(
#                 "Minimum key-term coverage", 0.0, 1.0, 0.10, 0.01
#             )
#         with c2:
#             min_claim_cov = st.slider(
#                 "Minimum claims-term coverage", 0.0, 1.0, 0.05, 0.01
#             )
#         with c3:
#             min_terms_matched = st.slider(
#                 "Minimum distinct key terms matched", 0, 10, 2, 1
#             )

#         c4, c5 = st.columns(2)
#         with c4:
#             min_feat_cov = st.slider(
#                 "Minimum feature coverage", 0.0, 1.0, 0.20, 0.01
#             )
#         with c5:
#             sem_z = st.slider(
#                 "Semantic outlier cutoff", 0.5, 4.0, 2.0, 0.1
#             )

#         known_file = st.file_uploader(
#             "Known relevant patents (optional)",
#             type=["xlsx", "csv"],
#             key="known_relevant_file",
#         )

#         submitted = st.form_submit_button(
#             "Search Patents",
#             type="primary",
#             use_container_width=True,
#         )

#     if submitted:
#         if not product_description.strip():
#             st.error("Please enter a product description.")
#             return

#         if uploaded_file is None:
#             st.error("Please upload the patent corpus.")
#             return

#         try:
#             corpus_path = _persist_upload(uploaded_file)
#             known_path = _persist_upload(known_file) if known_file is not None else None

#             with st.spinner("Reading and preparing patent corpus..."):
#                 prepared_df, corpus_key = load_prepared_corpus(corpus_path)

#             with st.spinner(
#                 "Running initial retrieval. The first semantic search may need to embed the corpus; "
#                 "subsequent searches reuse cached embeddings for 24 hours."
#             ):
#                 raw, metadata = retrieve_patents(
#                     prepared_df=prepared_df,
#                     corpus_key=corpus_key,
#                     product_description=product_description,
#                     similarity_threshold=float(tfidf_thr),
#                     semantic_threshold=float(sem_thr),
#                     use_openai_embeddings=bool(use_emb),
#                     validate=bool(validate),
#                     feature_percentile=float(feature_pct),
#                 )

#             strict = {
#                 "min_pass_share": float(min_pass_share),
#                 "min_term_coverage": float(min_term_cov),
#                 "min_claim_term_coverage": float(min_claim_cov),
#                 "min_terms_matched": int(min_terms_matched),
#                 "min_feature_coverage": float(min_feat_cov),
#                 "semantic_z_cutoff": float(sem_z),
#             }

#             final, status, diag = finalize(raw, metadata, strict, known_path)

#             st.session_state["raw_candidates"] = raw
#             st.session_state["search_metadata"] = metadata
#             st.session_state["final_candidates"] = final
#             st.session_state["final_status"] = status
#             st.session_state["validation_diagnostics"] = diag
#             st.session_state["known_path"] = known_path
#             st.session_state["strict_settings"] = strict
#             st.session_state["corpus_name"] = uploaded_file.name

#             st.success(
#                 f"Initial retrieval returned {len(raw)} candidates; "
#                 f"{len(final)} remain after validation."
#             )

#         except Exception as exc:
#             st.error(f"Patent search failed: {exc}")

#     if "final_candidates" not in st.session_state:
#         return

#     raw = st.session_state["raw_candidates"]
#     final = st.session_state["final_candidates"]
#     metadata = st.session_state["search_metadata"]

#     st.divider()
#     st.subheader("Search Summary")

#     c1, c2, c3 = st.columns(3)
#     c1.metric("Corpus records searched", metadata["corpus_size"])
#     c2.metric("Initial retrieval candidates", len(raw))
#     c3.metric("Final validated patents", len(final))

#     semantic_text = (
#         f"Semantic ≥ {metadata['semantic_threshold']:.2f}"
#         if metadata["semantic_search_used"]
#         else "Semantic retrieval disabled"
#     )

#     st.caption(
#         f"Corpus: {st.session_state.get('corpus_name', 'Uploaded corpus')} | "
#         f"Initial inclusion: TF-IDF ≥ {metadata['similarity_threshold']:.2f} "
#         f"OR {semantic_text} | "
#         f"Ranking: RRF, k=60 | Methods: {', '.join(metadata['methods_used'])}"
#     )

#     st.markdown(st.session_state["final_status"])

#     tab_final, tab_initial, tab_diagnostics = st.tabs(
#         ["Final Validated Results", "Initial Retrieval Results", "Validation Diagnostics"]
#     )

#     with tab_final:
#         if final.empty:
#             st.warning("No patents remain after validation.")
#         else:
#             columns = _display_columns(final)
#             st.dataframe(final[columns], use_container_width=True, hide_index=True)

#             st.download_button(
#                 "Download Final Results CSV",
#                 data=final[columns].to_csv(index=False).encode("utf-8-sig"),
#                 file_name="RelevanceIQ_Patent_Search_Results.csv",
#                 mime="text/csv",
#             )

#     with tab_initial:
#         st.caption(
#             "These are the original retrieval candidates before validation. "
#             "Their original Search Score and Search Rank are retained."
#         )

#         if raw.empty:
#             st.warning("The original retrieval method returned no candidates.")
#         else:
#             columns = _display_columns(raw)
#             st.dataframe(raw[columns], use_container_width=True, hide_index=True)

#             st.download_button(
#                 "Download Initial Retrieval CSV",
#                 data=raw[columns].to_csv(index=False).encode("utf-8-sig"),
#                 file_name="RelevanceIQ_Initial_Retrieval_Results.csv",
#                 mime="text/csv",
#             )

#     with tab_diagnostics:
#         diagnostics = st.session_state.get("validation_diagnostics", "")
#         if diagnostics:
#             st.markdown(diagnostics)
#         else:
#             st.info("No validation diagnostics are available.")

#     with st.expander("Inspect Patent Details"):
#         detail_source = final if not final.empty else raw

#         if not detail_source.empty:
#             options = [
#                 f"{row[PUBLICATION_COL]} — {row[TITLE_COL]}"
#                 for _, row in detail_source.iterrows()
#             ]

#             selected = st.selectbox("Select a patent", options)
#             row = detail_source.iloc[options.index(selected)]

#             st.markdown("**Abstract**")
#             st.write(row.get(ABSTRACT_COL, "") or "No abstract provided.")

#             st.markdown("**Claims**")
#             st.write(row.get(CLAIMS_COL, "") or "No claims provided.")

#             st.markdown("**Description**")
#             st.write(row.get(DESCRIPTION_COL, "") or "No description provided.")


# if __name__ == "__main__":
#     main()
