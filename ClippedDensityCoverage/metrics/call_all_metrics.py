from time import time
import torch

from top_pr import compute_top_pr as TopPR
from ClippedDensityCoverage.metrics.older_metrics_faster_code import (
    prdc,
    precision_recall_cover_fast,
)
from ClippedDensityCoverage.metrics.older_metrics_initial_code.call_alphaPrecision_betaRecall import (
    compute_initial_alpha_beta,
)
from ClippedDensityCoverage.metrics.Clipped_Density_Coverage import (
    ClippedDensityCoverage,
)

# Initial implementations
from ClippedDensityCoverage.metrics.older_metrics_initial_code.INITIAL_pp_pr import (
    compute_pprecision_precall,
)
from ClippedDensityCoverage.metrics.older_metrics_initial_code.INITIAL_prdc import (
    compute_prdc as old_prdc,
)
from ClippedDensityCoverage.metrics.older_metrics_initial_code.INITIAL_ASSYM_PR_manifmetric import (
    ManifoldMetric as old_ASSYM_PR,
)
from ClippedDensityCoverage.metrics.older_metrics_initial_code.INITIAL_PRC_from_pseudocode import (
    precision_recall_cover as old_PRC,
)


def call_evaluate_all(
    real_data,
    synthetic_data,
    metrics_of_interest=None,
    nearest_k=5,
    n_jobs=8,
    timing=False,
    print_results=False,
):
    res = {}

    if any(
        metric in metrics_of_interest
        for metric in [
            "Precision",
            "Recall",
            "Density",
            "Coverage",
            "symPrecision",
            "symRecall",
            "P-precision",
            "P-recall",
        ]
    ):
        start_time = time() if timing else None
        res_prdc = prdc(
            real_features=real_data,
            fake_features=synthetic_data,
            nearest_k=nearest_k,
            n_jobs=n_jobs,
            sym=any(
                metric in metrics_of_interest
                for metric in ["symPrecision", "symRecall"]
            ),
            ppr=any(
                metric in metrics_of_interest
                for metric in ["P-precision", "P-recall"]
            ),
        )
        if timing:
            print(f"prdc time: {time() - start_time:.4f} seconds")
        if print_results:
            for key, value in res_prdc.items():
                print(f"  {key}: {value}")
        res.update(res_prdc)

    if "PrecisionCover" in metrics_of_interest:
        start_time = time() if timing else None
        precision_cover = precision_recall_cover_fast(
            P=real_data,
            Q=synthetic_data,
        )
        res["PrecisionCover"] = precision_cover
        if timing:
            print(f"Precision Cover time: {time() - start_time:.4f} seconds")
        if print_results:
            print(f"  PrecisionCover: {precision_cover}")

    if "RecallCover" in metrics_of_interest:
        start_time = time() if timing else None
        recall_cover = precision_recall_cover_fast(
            P=synthetic_data,
            Q=real_data,
        )
        res["RecallCover"] = recall_cover
        if timing:
            print(f"Recall Cover time: {time() - start_time:.4f} seconds")
        if print_results:
            print(f"  RecallCover: {recall_cover}")

    if any(
        metric in metrics_of_interest
        for metric in [
            "ClippedDensity",
            "ClippedCoverage",
        ]
    ):
        start_time = time() if timing else None
        CDC = ClippedDensityCoverage(
            real_data=real_data,
            K=nearest_k,
            n_jobs=n_jobs,
        )
        if "ClippedDensity" in metrics_of_interest:
            res["ClippedDensity"] = CDC.ClippedDensity(
                synthetic_data=synthetic_data
            )
        if "ClippedCoverage" in metrics_of_interest:
            res["ClippedCoverage"] = CDC.ClippedCoverage(
                synthetic_data=synthetic_data
            )
        if timing:
            print(f"new_metrics time: {time() - start_time:.4f} seconds")
        if print_results:
            if "ClippedDensity" in metrics_of_interest:
                print(f"  ClippedDensity: {res['ClippedDensity']}")
            if "ClippedCoverage" in metrics_of_interest:
                print(f"  ClippedCoverage: {res['ClippedCoverage']}")

    if (
        "alpha-Precision" in metrics_of_interest
        or "beta-Recall" in metrics_of_interest
    ):
        start_time = time() if timing else None
        initial_alpha_precision, initial_beta_recall = (
            compute_initial_alpha_beta(
                real_data=real_data,
                synthetic_data=synthetic_data,
            )
        )
        res["alpha-Precision"] = initial_alpha_precision
        res["beta-Recall"] = initial_beta_recall
        if timing:
            print(f"alpha-beta time: {time() - start_time:.4f} seconds")
        if print_results:
            print(f"  alpha-Precision: {initial_alpha_precision}")
            print(f"  beta-Recall: {initial_beta_recall}")

    if any(metric in metrics_of_interest for metric in ["TopP", "TopR"]):
        start_time = time() if timing else None
        Top_PR = TopPR(
            real_features=real_data.reshape(real_data.shape[0], -1),
            fake_features=synthetic_data.reshape(synthetic_data.shape[0], -1),
            alpha=0.1,
            kernel="cosine",
            random_proj=True,
            f1_score=True,
        )

        Top_P = Top_PR.get("fidelity")
        Top_R = Top_PR.get("diversity")

        res["TopP"] = Top_P
        res["TopR"] = Top_R
        if timing:
            print(f"TopPR time: {time() - start_time:.4f} seconds")
        if print_results:
            print(f"  TopP: {Top_P}")
            print(f"  TopR: {Top_R}")

    # OLD IMPLEMENTATIONS
    if any(
        metric in metrics_of_interest
        for metric in [
            "oldPrecision",
            "oldRecall",
            "oldDensity",
            "oldCoverage",
        ]
    ):
        start_time = time() if timing else None
        res_prdc = old_prdc(
            real_features=real_data.reshape(real_data.shape[0], -1),
            fake_features=synthetic_data.reshape(synthetic_data.shape[0], -1),
            nearest_k=nearest_k,
        )
        if timing:
            print(f"old prdc time: {time() - start_time:.4f} seconds")
        if print_results:
            print(f"  oldPrecision: {res_prdc['precision']}")
            print(f"  oldRecall: {res_prdc['recall']}")
            print(f"  oldDensity: {res_prdc['density']}")
            print(f"  oldCoverage: {res_prdc['coverage']}")

        res["oldPrecision"] = res_prdc["precision"]
        res["oldRecall"] = res_prdc["recall"]
        res["oldDensity"] = res_prdc["density"]
        res["oldCoverage"] = res_prdc["coverage"]

    if any(
        metric in metrics_of_interest
        for metric in [
            "oldSymPrecision",
            "oldSymRecall",
        ]
    ):
        start_time = time() if timing else None

        assym = old_ASSYM_PR(
            ref_data=real_data, gen_data=synthetic_data, k=nearest_k
        )
        old_sym_precision = assym.sym_precision()
        res["oldSymPrecision"] = old_sym_precision["sym_precision"]
        print("Precision from symPrecision:", old_sym_precision["precision"])

        old_sym_recall = assym.sym_recall()
        res["oldSymRecall"] = old_sym_recall["sym_recall"]
        print("Recall from symRecall:", old_sym_recall["recall"])
        print("Coverage from symRecall:", old_sym_recall["coverage"])
        if timing:
            print(f"old symPrecision time: {time() - start_time:.4f} seconds")
        if print_results:
            print(f"  oldSymPrecision: {old_sym_precision['sym_precision']}")
            print(f"  oldSymRecall: {old_sym_recall['sym_recall']}")

    if any(
        metric in metrics_of_interest
        for metric in ["oldPprecision", "oldPrecall"]
    ):
        start_time = time() if timing else None
        pp_precision, pp_recall = compute_pprecision_precall(
            real=real_data.reshape(real_data.shape[0], -1),
            fake=synthetic_data.reshape(synthetic_data.shape[0], -1),
            kth=nearest_k,
            gpu=torch.cuda.is_available(),
        )

        res["oldPprecision"] = pp_precision
        res["oldPrecall"] = pp_recall
        if timing:
            print(
                f"Pprecision Precall time: {time() - start_time:.4f} seconds"
            )
        if print_results:
            print(f"  oldPprecision: {pp_precision}")
            print(f"  oldPrecall: {pp_recall}")

    if "oldPrecisionCover" in metrics_of_interest:
        start_time = time() if timing else None

        old_precision_cover = old_PRC(
            P=real_data.reshape(real_data.shape[0], -1),
            Q=synthetic_data.reshape(synthetic_data.shape[0], -1),
        )
        res["oldPrecisionCover"] = old_precision_cover

        if timing:
            print(
                f"old Precision Cover time: {time() - start_time:.4f} seconds"
            )
        if print_results:
            print(f"  oldPrecisionCover: {old_precision_cover}")

    if "oldRecallCover" in metrics_of_interest:
        start_time = time() if timing else None

        old_recall_cover = old_PRC(
            P=synthetic_data.reshape(synthetic_data.shape[0], -1),
            Q=real_data.reshape(real_data.shape[0], -1),
        )
        res["oldRecallCover"] = old_recall_cover

        if timing:
            print(f"old Recall Cover time: {time() - start_time:.4f} seconds")
        if print_results:
            print(f"  oldRecallCover: {old_recall_cover}")
    return res
