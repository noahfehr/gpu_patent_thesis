import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import tomotopy as tp
import itertools
import subprocess
import tempfile
import os
from collections import Counter
# Implemented based on stability metrics outlined here: https://arxiv.org/pdf/2410.23186v2
# -----------------------------
# HELPERS (match repo behavior)
# -----------------------------
def row_normalize(A, eps=1e-12):
    A = np.asarray(A)
    return A / (A.sum(axis=1, keepdims=True) + eps)

def cosine(u, v, eps=1e-12):
    u = np.asarray(u); v = np.asarray(v)
    denom = (np.linalg.norm(u) * np.linalg.norm(v)) + eps
    return float(np.dot(u, v) / denom)

def greedy_topic_alignment_cosine(post_words_run, post_words_ref):
    """
    Greedy one-to-one matching to replication 1 using cosine similarity 
    of topic-word probability vectors (nested-loop greedy, not Hungarian).

    for each topic in the run (top_idx),
      for each topic in reference (top_idx2),
        compute cosine(run_topic, ref_topic)
      take best ref topic not yet used; store in new_top_ordering[top_idx]

    Returns:
      new_top_ordering: length K, entries are reference-topic indices
    """
    K = post_words_ref.shape[0]
    new_top_ordering = [-1] * K
    max_cos_vals = [-1.0] * K

    for top_idx in range(K):
        for top_idx2 in range(K):
            new_max = cosine(post_words_run[top_idx, :], post_words_ref[top_idx2, :])
            corresponding_idx = top_idx2
            if (new_max > max_cos_vals[top_idx]) and (corresponding_idx not in new_top_ordering):
                max_cos_vals[top_idx] = new_max
                new_top_ordering[top_idx] = corresponding_idx

    # Safety: fill any remaining -1 (should be rare) with unused refs
    used = set([i for i in new_top_ordering if i != -1])
    unused = [i for i in range(K) if i not in used]
    for i in range(K):
        if new_top_ordering[i] == -1:
            new_top_ordering[i] = unused.pop(0)

    return np.array(new_top_ordering, dtype=int)

def omega_psych_via_r(X):
    """
    Calls R psych::omega exactly as in tm_reliab.R:
      suppressMessages(suppressWarnings(omega(data.frame(X), nfactors=1)))
      returns omega.tot and mean(stats$sd) (used for their OmegaSE)

    X is a 2D numpy array: rows are observations, cols are replications (items).
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] < 2:
        return (np.nan, np.nan)

    # Write X to a temp CSV without headers (R will read.matrix)
    with tempfile.TemporaryDirectory() as td:
        csv_path = os.path.join(td, "x.csv")
        np.savetxt(csv_path, X, delimiter=",")

        r_code = r"""
        suppressMessages(suppressWarnings(library(psych)))
        x <- as.matrix(read.csv("%s", header=FALSE))
        # match repo: omega(data.frame(do.call("cbind", ...)), nfactors=1)
        # here x already is cbind'ed matrix across replications
        res <- suppressMessages(suppressWarnings(omega(as.data.frame(x), nfactors=1)))
        omega_tot <- res$omega.tot
        # repo uses omega(... )$stats$sd and then mean() across topics
        # stats$sd is a vector; return its mean as an analogue to their usage
        sd_mean <- mean(res$stats$sd)
        cat(sprintf("%%.17g,%%.17g\n", omega_tot, sd_mean))
        """ % (csv_path.replace("\\", "/"))

        proc = subprocess.run(
            ["Rscript", "-e", r_code],
            capture_output=True,
            text=True
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "R psych::omega call failed.\n"
                f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
            )

        out = proc.stdout.strip().splitlines()[-1]
        omega_tot_str, sd_mean_str = out.split(",")
        return (float(omega_tot_str), float(sd_mean_str))

def _extract_topic_word_matrix(model, vocab_index):
    """
    Extract topic-word probabilities into a fixed global vocabulary space.

    Parameters
    ----------
    model : tp.LDAModel
    vocab_index : dict[str, int]
        Global token -> column index mapping.

    Returns
    -------
    H : np.ndarray of shape (K, V)
        Topic-word matrix in the fixed vocabulary space.
    """
    K = model.k
    V = len(vocab_index)
    H = np.zeros((K, V), dtype=float)

    model_vocab = list(model.used_vocabs)
    if len(model_vocab) == 0:
        return H

    model_to_global = []
    for word in model_vocab:
        model_to_global.append(vocab_index.get(word, -1))

    for k in range(K):
        topic_dist = np.asarray(model.get_topic_word_dist(k), dtype=float)
        for j, gidx in enumerate(model_to_global):
            if gidx != -1:
                H[k, gidx] = topic_dist[j]

    return H

def _build_fixed_vocab(tokenized_docs, min_df=1):
    """
    Build a fixed corpus-wide vocabulary using document frequency.

    Parameters
    ----------
    tokenized_docs : list[list[str]]
        Tokenized documents.
    min_df : int
        Keep tokens appearing in at least min_df documents.

    Returns
    -------
    vocab_list : list[str]
        Fixed vocab in deterministic order.
    vocab_index : dict[str, int]
        Token -> column index.
    """
    df_counter = Counter()
    for doc in tokenized_docs:
        for tok in set(doc):
            df_counter[tok] += 1

    vocab_list = sorted([tok for tok, df in df_counter.items() if df >= min_df])
    vocab_index = {tok: i for i, tok in enumerate(vocab_list)}
    return vocab_list, vocab_index
# -----------------------------
# MAIN FUNCTION
# -----------------------------
def _build_model(n_topics, seed, **model_kwargs):
    return tp.LDAModel(
        k=n_topics,
        seed=seed,
        alpha=0.3,
        eta=0.05,
        **model_kwargs
    )

def _fit_and_extract(model, tokenized_docs, vocab_index, iterations=100):
    """
    Fit a tomotopy LDA model and extract W and H.

    Parameters
    ----------
    model : tp.LDAModel
    tokenized_docs : list[list[str]]
    vocab_index : dict[str, int]
    iterations : int

    Returns
    -------
    W : np.ndarray
        Document-topic matrix, shape (D, K)
    H : np.ndarray
        Topic-word matrix, shape (K, V), in fixed vocab space
    """
    for doc in tokenized_docs:
        if doc:
            model.add_doc(doc)

    model.train(iterations)

    W = np.array([doc.get_topic_dist() for doc in model.docs], dtype=float)
    H = _extract_topic_word_matrix(model, vocab_index)

    return W, H

def compute_topic_stability(tokenized_docs, k_values, seeds, iterations=100, vocab_min_df=1, **model_kwargs):
    """
    Compute LDA topic model stability across different k values and random seeds.

    Parameters
    ----------
    tokenized_docs : list[list[str]]
        Tokenized documents used to fit the model.
    k_values : list[int]
        Numbers of topics to evaluate.
    seeds : list[int]
        Random seeds for repeated fits.
    iterations : int, default=100
        Number of Gibbs sampling iterations per fit.
    vocab_min_df : int, default=1
        Minimum corpus document frequency for the fixed comparison vocabulary.
    model_kwargs :
        Additional kwargs passed to tp.LDAModel.

    Returns
    -------
    pd.DataFrame
        Stability metrics for each k.
    """
    results = []

    # Fixed vocab for cross-run comparability
    vocab_list, vocab_index = _build_fixed_vocab(tokenized_docs, min_df=vocab_min_df)
    print(f"Fixed comparison vocab size: {len(vocab_list)}")

    for n_topics in k_values:
        print(f"\n===== k = {n_topics} =====")

        redundancy_scores = []
        thetas = []
        post_words_list = []
        post_words_ref = None

        for seed in seeds:
            model = _build_model(n_topics, seed, **model_kwargs)
            W, H = _fit_and_extract(
                model=model,
                tokenized_docs=tokenized_docs,
                vocab_index=vocab_index,
                iterations=iterations
            )

            H_prob = row_normalize(H)
            S = cosine_similarity(H_prob)
            np.fill_diagonal(S, np.nan)
            redundancy_scores.append({
                "max_sim": np.nanmax(S),
                "mean_sim": np.nanmean(S)
            })

            theta = row_normalize(W)
            post_words = row_normalize(H)

            if post_words_ref is None:
                post_words_ref = post_words.copy()
                thetas.append(theta)
                post_words_list.append(post_words)
            else:
                new_top_ordering = greedy_topic_alignment_cosine(post_words, post_words_ref)
                theta_aligned = theta[:, new_top_ordering]
                post_words_aligned = post_words[new_top_ordering, :]
                thetas.append(theta_aligned)
                post_words_list.append(post_words_aligned)

        K_eff = n_topics - 1
        omega1_vals = []
        omega2_vals = []
        omega1_sd_means = []
        omega2_sd_means = []

        for t in range(K_eff):
            X_theta_t = np.column_stack([thetas[r][:, t] for r in range(len(seeds))])
            o1, sd1 = omega_psych_via_r(X_theta_t)
            omega1_vals.append(o1)
            omega1_sd_means.append(sd1)

            X_words_t = np.column_stack([post_words_list[r][t, :] for r in range(len(seeds))])
            o2, sd2 = omega_psych_via_r(X_words_t)
            omega2_vals.append(o2)
            omega2_sd_means.append(sd2)

        omega1 = float(np.mean(omega1_vals))
        omega2 = float(np.mean(omega2_vals))
        omega_val = (omega1 + omega2) / 2.0
        omega_se = float(
            np.mean([np.mean(omega1_sd_means), np.mean(omega2_sd_means)])
            * np.sqrt((1.0 / K_eff) + (1.0 / K_eff))
        )

        avg_max_sim = np.mean([r["max_sim"] for r in redundancy_scores])
        avg_mean_sim = np.mean([r["mean_sim"] for r in redundancy_scores])

        print(f"Omega (theta-side):      {omega1:.3f}")
        print(f"Omega (words-side):      {omega2:.3f}")
        print(f"Omega (avg):             {omega_val:.3f}")
        print(f"Omega SE (repo-style):   {omega_se:.4f}")
        print(f"Avg Max Topic Cosine:    {avg_max_sim:.3f}")
        print(f"Avg Mean Topic Cosine:   {avg_mean_sim:.3f}")

        results.append({
            "k": n_topics,
            "omega_theta": omega1,
            "omega_words": omega2,
            "omega": omega_val,
            "omega_se": omega_se,
            "max_cosine": float(avg_max_sim),
            "mean_cosine": float(avg_mean_sim),
        })

    return pd.DataFrame(results)