import numpy as np
from sklearn.neighbors import NearestNeighbors


def prdc(
    real_features,
    fake_features,
    nearest_k,
    sym=False,
    ppr=False,
    a_ppr=1.2,
    n_jobs=8,
):
    """
    Computes precision, recall, density, coverage, and optionally symmetric
    precision & recall and P-precision & P-recall.

    Args:
        real_features (np.ndarray): real features
        fake_features (np.ndarray): fake features
        nearest_k (int): number of nearest neighbors to consider
        sym (bool, optional): whether to compute symmetric precision and
            recall. Defaults to False.
        ppr (bool, optional): whether to compute P-precision and P-recall.
            Defaults to False.
        n_jobs (int, optional): number of jobs to run in parallel. Defaults to
            8.

    Returns:
        dictionary containing precision, recall, density, coverage and
        optionally symmetric precision and recall.
    """
    N = real_features.shape[0]
    M = fake_features.shape[0]
    real_features_flat = real_features.reshape(N, -1)
    fake_features_flat = fake_features.reshape(M, -1)

    real_NN = NearestNeighbors(
        n_neighbors=nearest_k + 1,
        n_jobs=n_jobs,
    ).fit(real_features_flat)
    distances_real_to_real, _ = real_NN.kneighbors(real_features_flat)
    radii_real = distances_real_to_real[:, -1]

    fake_NN = NearestNeighbors(
        n_neighbors=nearest_k + 1,
        n_jobs=n_jobs,
    ).fit(fake_features_flat)
    distances_fake_to_fake, _ = fake_NN.kneighbors(fake_features_flat)
    radii_fake = distances_fake_to_fake[:, -1]

    low_parallel = n_jobs <= 4
    if low_parallel:
        fake_tree = NearestNeighbors(
            n_neighbors=nearest_k,
            algorithm="ball_tree",
            n_jobs=1,  # to allow varying radii
        ).fit(fake_features_flat)
        all_fake_in_real_balls = fake_tree.radius_neighbors(
            real_features_flat, radius=radii_real, return_distance=False
        )
    else:
        # Use parallel computation anyways
        fake_tree = NearestNeighbors(
            n_neighbors=nearest_k,
            algorithm="ball_tree",
            n_jobs=n_jobs,
        ).fit(fake_features_flat)
        distances_candidates, all_fake_in_real_balls_candidates = (
            fake_tree.radius_neighbors(
                real_features_flat,
                radius=np.max(radii_real),
                return_distance=True,
            )
        )
        all_fake_in_real_balls = np.empty(N, dtype=np.ndarray)
        for i in range(N):
            # Filter out points that are too far away
            valid_indices = distances_candidates[i] <= radii_real[i]
            all_fake_in_real_balls[i] = all_fake_in_real_balls_candidates[i][
                valid_indices
            ]

    # Precision: is each fake point inside a real ball?
    # radii_real are the radii of the real balls
    # all_fake_in_real_balls is the list of fake points inside each real ball
    fake_appearance_count = np.bincount(
        np.concatenate(all_fake_in_real_balls), minlength=M
    )
    # fake_appearance_count[i] is the number of real balls the fake point
    # i is in
    precision = np.mean(fake_appearance_count > 0)

    # Density is the average number of real balls each fake point is in
    density = np.mean(fake_appearance_count) / nearest_k

    # Coverage: is there a fake point inside each real ball?
    coverage = np.mean(
        [
            len(fake_in_real_ball_i) > 0
            for fake_in_real_ball_i in all_fake_in_real_balls
        ]
    )

    if low_parallel:
        real_tree = NearestNeighbors(
            n_neighbors=nearest_k,
            algorithm="ball_tree",
            n_jobs=1,  # to allow varying radii
        ).fit(real_features_flat)
        all_real_in_fake_balls = real_tree.radius_neighbors(
            fake_features_flat, radius=radii_fake, return_distance=False
        )
    else:
        # Use parallel computation anyways
        real_tree = NearestNeighbors(
            n_neighbors=nearest_k,
            algorithm="ball_tree",
            n_jobs=n_jobs,
        ).fit(real_features_flat)
        distances_candidates, all_real_in_fake_balls_candidates = (
            real_tree.radius_neighbors(
                fake_features_flat,
                radius=np.max(radii_fake),
                return_distance=True,
            )
        )
        all_real_in_fake_balls = np.empty(M, dtype=np.ndarray)
        for j in range(M):
            # Filter out points that are too far away
            valid_indices = distances_candidates[j] <= radii_fake[j]
            all_real_in_fake_balls[j] = all_real_in_fake_balls_candidates[j][
                valid_indices
            ]
    # Recall: is each real point inside a fake ball?
    # radii_fake are the radii of the fake balls
    # all_real_in_fake_balls is the list of real points inside each fake ball
    real_appearance_count = np.bincount(
        np.concatenate(all_real_in_fake_balls), minlength=N
    )
    # real_appearance_count[i] is the number of fake balls the real point i is
    # in
    recall = np.mean(real_appearance_count > 0)

    res = dict(
        Precision=precision,
        Recall=recall,
        Density=density,
        Coverage=coverage,
    )

    if sym:
        symRecall = np.min((recall, coverage))

        # cPrecision is the symmetric of coverage: is there a real point in
        # each fake ball?
        cPrecision = np.mean(
            [
                len(real_in_fake_ball_i) > 0
                for real_in_fake_ball_i in all_real_in_fake_balls
            ]
        )

        symPrecision = np.min((precision, cPrecision))

        res["symRecall"] = symRecall
        res["symPrecision"] = symPrecision

    if ppr:
        R_precision = np.mean(radii_real) * a_ppr
        R_recall = np.mean(radii_fake) * a_ppr

        if low_parallel and n_jobs > 1:
            # Can be done in parallel because the radius is the same for all
            # points, so build the trees again with n_jobs>1
            fake_tree = NearestNeighbors(
                n_neighbors=nearest_k,
                algorithm="ball_tree",
                n_jobs=n_jobs,
            ).fit(fake_features_flat)
            real_tree = NearestNeighbors(
                n_neighbors=nearest_k,
                algorithm="ball_tree",
                n_jobs=n_jobs,
            ).fit(real_features_flat)

        distances_real_in_fake_balls, all_real_in_fake_balls = (
            real_tree.radius_neighbors(
                fake_features_flat,
                radius=R_precision,
                return_distance=True,
            )
        )
        prods = np.ones(M)
        for j in range(M):
            if len(distances_real_in_fake_balls[j]) > 0:
                prods[j] = np.prod(
                    distances_real_in_fake_balls[j] / R_precision
                )

        # P-precision is the average over all fake points of the product of
        # the probability over real points that y_j does not belong to the
        # support around x_i
        res["P-precision"] = np.mean(1 - prods)

        distances_fake_in_real_balls, all_fake_in_real_balls = (
            fake_tree.radius_neighbors(
                real_features_flat,
                radius=R_recall,
                return_distance=True,
            )
        )
        prods = np.ones(N)
        for i in range(N):
            if len(distances_fake_in_real_balls[i]) > 0:
                prods[i] = np.prod(distances_fake_in_real_balls[i] / R_recall)
        res["P-recall"] = np.mean(1 - prods)

    return res


def precision_recall_cover_fast(P, Q, k=3, C=3, n_jobs=22):
    # Different function because k and kprime are not the same as above.

    P_flat = P.reshape(P.shape[0], -1)
    Q_flat = Q.reshape(Q.shape[0], -1)

    k_prime = C * k

    low_parallel = n_jobs <= 4
    if low_parallel:
        P_tree = NearestNeighbors(
            n_neighbors=k, algorithm="ball_tree", n_jobs=1
        ).fit(P_flat)
        # When a tree is queried with radius_neighbors, it returns the indices
        # of the neighbors in the fitted data set that are within the specified
        # radius of the query point.

        # queries with varying radius do not work in parallel, hence we use
        # n_jobs=1
    else:
        P_tree = NearestNeighbors(
            n_neighbors=k, algorithm="ball_tree", n_jobs=n_jobs
        ).fit(P_flat)

    # Compute k' nearest neighbors in Q for each point in Q
    Q_neighbors = NearestNeighbors(n_neighbors=k_prime, n_jobs=n_jobs).fit(
        Q_flat
    )
    distances_k_prime, _ = Q_neighbors.kneighbors(Q_flat)
    r_Q = distances_k_prime[:, -1]

    # For each point in Q, find the indices of the points in P that are within
    # the corresponding radius r_Q
    if low_parallel:
        indices = P_tree.radius_neighbors(
            Q_flat, radius=r_Q, return_distance=False
        )
    else:
        dists, indices_candidates = P_tree.radius_neighbors(
            Q_flat, radius=np.max(r_Q), return_distance=True
        )
        indices = [
            idx[dists[i] <= r_Q[i]] for i, idx in enumerate(indices_candidates)
        ]

    is_covered = np.array([len(idx) >= k for idx in indices])
    covered_count = np.sum(is_covered)

    PC = covered_count / len(Q_flat)
    return PC
