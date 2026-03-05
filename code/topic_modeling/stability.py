import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
import itertools
import subprocess
import tempfile
import os
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

# -----------------------------
# MAIN FUNCTION
# -----------------------------
def _build_model(method, n_topics, seed, **model_kwargs):
    """Factory function to build NMF or LDA model."""
    if method == 'nmf':
        return NMF(
            n_components=n_topics,
            init="nndsvda",
            random_state=seed,
            max_iter=500,
            alpha_W=0,
            alpha_H=0.3,
            **model_kwargs
        )
    elif method == 'lda':
        return LatentDirichletAllocation(
            n_components=n_topics,
            random_state=seed,
            learning_method="batch",
            max_iter=100,
            doc_topic_prior=0.3,
            topic_word_prior=0.05,
            **model_kwargs
        )

def _fit_and_extract(model, X, method):
    """Fit model and extract W (doc-topic) and H (topic-word) matrices."""
    if method == 'nmf':
        W = model.fit_transform(X)
        H = model.components_
        recon_err = model.reconstruction_err_
    elif method == 'lda':
        W = model.fit_transform(X)
        H = np.exp(model.components_)  # Convert from log-space
        recon_err = None
    return W, H, recon_err

def compute_topic_stability(k_values, seeds, X, method='nmf', **model_kwargs):
    """
    Compute topic model stability across different k values and random seeds.

    Parameters:
    - k_values: list of int, number of topics to test
    - seeds: list of int, random seeds for reproducibility
    - X: sparse matrix, document-term matrix (TF-IDF for NMF, count for LDA)
    - method: str, 'nmf' or 'lda'
    - model_kwargs: additional kwargs for the model (NMF or LDA)

    Returns:
    - pd.DataFrame with stability metrics for each k
    """
    if method not in ['nmf', 'lda']:
        raise ValueError("method must be 'nmf' or 'lda'")

    results = []

    for n_topics in k_values:
        print(f"\n===== k = {n_topics} =====")

        reconstruction_errors = []
        redundancy_scores = []
        thetas = []
        post_words_list = []
        post_words_ref = None

        # Fit models across all seeds
        for seed in seeds:
            model = _build_model(method, n_topics, seed, **model_kwargs)
            W, H, recon_err = _fit_and_extract(model, X, method)

            if recon_err is not None:
                reconstruction_errors.append(recon_err)

            # Redundancy: compute cosine similarity of normalized topics
            H_prob = row_normalize(H)
            S = cosine_similarity(H_prob)
            np.fill_diagonal(S, np.nan)
            redundancy_scores.append({
                "max_sim": np.nanmax(S),
                "mean_sim": np.nanmean(S)
            })

            # Construct theta and post_words
            theta = row_normalize(W)
            post_words = row_normalize(H)

            # Align to first replication using greedy cosine matching
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

        # Compute stability via multidimensional Omega
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

        # Print results
        if reconstruction_errors:
            print(f"Avg Reconstruction Error: {np.mean(reconstruction_errors):.4f}")
        print(f"Omega (theta-side):      {omega1:.3f}")
        print(f"Omega (words-side):      {omega2:.3f}")
        print(f"Omega (avg):             {omega_val:.3f}")
        print(f"Omega SE (repo-style):   {omega_se:.4f}")
        print(f"Avg Max Topic Cosine:    {avg_max_sim:.3f}")
        print(f"Avg Mean Topic Cosine:   {avg_mean_sim:.3f}")

        # Store results
        result_dict = {
            "k": n_topics,
            "omega_theta": omega1,
            "omega_words": omega2,
            "omega": omega_val,
            "omega_se": omega_se,
            "max_cosine": float(avg_max_sim),
            "mean_cosine": float(avg_mean_sim),
        }
        if reconstruction_errors:
            result_dict["recon_error"] = float(np.mean(reconstruction_errors))
        
        results.append(result_dict)

    results_df = pd.DataFrame(results)
    return results_df