def doc_to_ijv(documents):
    """
    Converts the input documents into triplet (IJV) format.

    Parameters
    ----------
    documents: list of 2D arrays
        Each document is represented as a 2-row matrix:
        - First row: Term indices
        - Second row: Term counts.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - 'i': List of document indices.
        - 'j': List of term indices.
        - 'k': List of term counts.
        - 'rowsums': List of total token counts per document.
    """
    indices = [doc[0] for doc in documents] # Grab the first row (term indices)
    counts = [doc[1] for doc in documents] # Grab the second row (term counts)

    v_sub_d = [len(term_counts) for term_counts in counts] # Number of unique terms per document
    row_sums = [sum(term_counts) for term_counts in counts] # Total tokens per document

    # Flatten indices and counts, and repeat document IDs
    i = [doc_id for doc_id, term_count in enumerate(v_sub_d) for _ in range(term_count)]
    j = [index for term_indices in indices for index in term_indices]
    v = [count for term_counts in counts for count in term_counts]

    return {
        "i": i,
        "j": j,
        "v": v,
        "rowsums": row_sums
    }