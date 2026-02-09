import torch
from ClippedDensityCoverage.metrics.older_metrics_initial_code.INITIAL_ALPHA_BETA_evaluation import (
    compute_alpha_precision,
)
from ClippedDensityCoverage.metrics.older_metrics_initial_code.INITIAL_ALPHA_BETA_representations.OneClass import (
    OneClassLayer,
)


def compute_initial_alpha_beta(real_data, synthetic_data, **kwargs):
    # The hyperparameters follow the values in Appendix D.4.2.
    # For the unspecified hyperparameters, we use the values from the notebook
    # toy_metric_evaluation.ipynb
    # https://github.com/vanderschaarlab/evaluating-generative-models/blob/main/toy_metric_evaluation.ipynb
    #
    # We tried those in main_image.py
    # https://github.com/vanderschaarlab/evaluating-generative-models/blob/main/main_image.py
    # for the image data and checked the performances on the validation set as
    # recommended. We got worse results, so we stuck with the hyperparameters
    # from the toy data. This is probably because we are using embeddings
    # instead of images.
    # The model has a very strong tendency to overfit, and works better with
    # 100 epochs than more, so we use the exact same parametrisation as for
    # the toy data.
    real_data_flatten = real_data.reshape(real_data.shape[0], -1)
    synthetic_data_flatten = synthetic_data.reshape(
        synthetic_data.shape[0], -1
    )

    data_type = "toy"  # Hardcoded because it works better.
    OC_params = {
        "rep_dim": (
            real_data_flatten.shape[1] if data_type == "toy" else 32
        ),  # set in appendix D.4.2 for image data to 32 but we have embeddings
        "num_layers": (
            2 if data_type == "toy" else 3
        ),  # set in appendix D.4.2 for image data
        "num_hidden": (
            200 if data_type == "toy" else 128
        ),  # set in appendix D.4.2 for image data
        "activation": "ReLU",  # explicitely set in appendix D.4.2
        "dropout_prob": 0.0,
        "dropout_active": False,  # In both cases
        "LossFn": "SoftBoundary",  # explicitely set in appendix D.4.2
        "lr": 1e-3 if data_type == "toy" else 2e-3,  # 1e-3 works better
        "epochs": 100 if data_type == "toy" else 2000,
        # initially 2000, but even with 300 epochs the model overfits a lot
        # -> 100 epochs
        "warm_up_epochs": 10,  # In both cases
        "train_prop": 1.0 if data_type == "toy" else 0.8,
        "weight_decay": 1e-2,  # In both cases
        "input_dim": real_data_flatten.shape[1],  # In both cases
    }

    OC_hyperparams = {
        "Radius": 1,  # In both cases
        "nu": 1e-2,  # explicitely set in appendix D.4.2
        "center": (
            torch.ones(real_data.shape[1])
            if data_type == "toy"
            else 10 * torch.ones(real_data.shape[1])
        ),  # explicitely set in appendix D.4.2,
        # 10 yields very poor performances for us
    }

    model = OneClassLayer(params=OC_params, hyperparams=OC_hyperparams)
    model.fit(real_data_flatten, verbosity=False)

    X_out = (
        model(torch.tensor(real_data_flatten).float().to("cuda"))
        .cpu()
        .float()
        .detach()
        .numpy()
    )
    Y_out = (
        model(torch.tensor(synthetic_data_flatten).float().to("cuda"))
        .cpu()
        .float()
        .detach()
        .numpy()
    )
    emb_center = model.c
    _, _, _, ini_alpha_prec, ini_beta_recall, _ = compute_alpha_precision(
        X_out, Y_out, emb_center
    )

    return ini_alpha_prec, ini_beta_recall
