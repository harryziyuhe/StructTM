from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, isspmatrix_csr, isspmatrix_coo
import pandas as pd


def as_stm_corpus(
    documents: Union[List[List[Tuple[int, int]]], csr_matrix, coo_matrix],
    vocab: Optional[List[str]] = None,
    data: Optional[pd.DataFrame] = None,
) -> Dict[str, Union[List, List[str], Optional[pd.DataFrame]]]:
    """
    Convert document-term counts and metadata into a standardized STM corpus format.

    Parameters
    ----------
    documents : list of list of tuple or scipy.sparse.csr_matrix or scipy.sparse.coo_matrix
        Document-term representation.
        - If a `list`, each document should be a list of (term_index, count) tuples.
        - If a `csr_matrix` or `coo_matrix`, it should be a sparse matrix where
          rows represent documents and columns represent terms.
    vocab : list of str, optional
        List of vocabulary terms. Required if `documents` is a list; ignored for sparse matrices.
    data : pandas.DataFrame, optional
        Metadata associated with the documents.

    Returns
    -------
    dict
        Dictionary with the following keys:
        - `documents`: Processed document-term counts in list format.
        - `vocab`: List of vocabulary terms.
        - `data`: Metadata provided in the `data` parameter.

    Raises
    ------
    ValueError
        If the input format is invalid or if required parameters are missing.

    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> docs = [[(0, 2), (1, 3)], [(1, 1), (2, 4)]]
    >>> vocab = ["term0", "term1", "term2"]
    >>> as_stm_corpus(docs, vocab=vocab)
    {'documents': [[(0, 2), (1, 3)], [(1, 1), (2, 4)]], 'vocab': ['term0', 'term1', 'term2'], 'data': None}

    >>> sparse_docs = csr_matrix([[2, 3, 0], [0, 1, 4]])
    >>> as_stm_corpus(sparse_docs)
    {'documents': [...], 'vocab': ['0', '1', '2'], 'data': None}
    """
    if isinstance(documents, list):
        return as_stm_corpus_list(documents, vocab, data)

    if isspmatrix_csr(documents):
        return as_stm_corpus_csr(documents, data)

    if isspmatrix_coo(documents):
        return as_stm_corpus_coo(documents, data)

    raise ValueError("Unsupported format for 'documents'. Must be list or sparse matrix.")


def as_stm_corpus_list(
    documents: List[List[Tuple[int, int]]],
    vocab: List[str],
    data: Optional[pd.DataFrame] = None,
) -> Dict[str, Union[List[List[Tuple[int, int]]], List[str], Optional[pd.DataFrame]]]:
    """
    Process documents in list format.

    Parameters
    ----------
    documents : list of list of tuple
        Each document is a list of (term_index, count) tuples.
    vocab : list of str
        List of vocabulary terms.
    data : pandas.DataFrame, optional
        Metadata associated with the documents.

    Returns
    -------
    dict
        Processed corpus dictionary.
    """
    if not isinstance(vocab, list):
        raise ValueError("Vocab must be a list of terms.")

    for doc in documents:
        if not all(isinstance(pair, tuple) and len(pair) == 2 for pair in doc):
            raise ValueError("Each document must contain (term_index, count) pairs.")

    return {
        "documents": documents,
        "vocab": vocab,
        "data": data,
    }


def as_stm_corpus_csr(
    documents: csr_matrix,
    data: Optional[pd.DataFrame] = None,
) -> Dict[str, Union[List[List[List[int]]], List[str], Optional[pd.DataFrame]]]:
    """
    Process documents in CSR matrix format.

    Parameters
    ----------
    documents : scipy.sparse.csr_matrix
        Document-term matrix where rows are documents and columns are terms.
    data : pandas.DataFrame, optional
        Metadata associated with the documents.

    Returns
    -------
    dict
        Processed corpus dictionary.
    """
    # Drop unused terms (columns with zero counts)
    nonzero_cols = documents.sum(axis=0).A1 > 0
    documents = documents[:, nonzero_cols]
    vocab = [str(i) for i in range(documents.shape[1])]

    # Convert to list of term counts
    doc_list = []
    for i in range(documents.shape[0]):
        row = documents[i].nonzero()
        counts = documents[i].data
        doc_list.append(np.vstack((row[1] + 1, counts)).tolist())  # +1 to match 1-indexing

    return {
        "documents": doc_list,
        "vocab": vocab,
        "data": data,
    }


def as_stm_corpus_coo(
    documents: coo_matrix,
    data: Optional[pd.DataFrame] = None,
) -> Dict[str, Union[List[List[List[int]]], List[str], Optional[pd.DataFrame]]]:
    """
    Process documents in COO matrix format.

    Parameters
    ----------
    documents : scipy.sparse.coo_matrix
        Document-term matrix where rows are documents and columns are terms.
    data : pandas.DataFrame, optional
        Metadata associated with the documents.

    Returns
    -------
    dict
        Processed corpus dictionary.
    """
    csr_docs = documents.tocsr()
    return as_stm_corpus_csr(csr_docs, data)
