"""
The whole INITIAL_ALPHA_BETA_representations folder is a copy of the
representations folder from the authors of alpha-Precision and beta-Recall,
taken from their github:
https://github.com/vanderschaarlab/evaluating-generative-models/tree/main/representations

Ahmed Alaa, Boris Van Breugel, Evgeny S Saveliev, and Mihaela van der Schaar.
How faithful is your synthetic data? sample-level metrics for evaluating and
auditing generative models. In International Conference on Machine Learning,
pages 290–306. PMLR, 2022.149

Under MIT license, see INITIAL_ALPHA_BETA_LICENSE.txt
"""

"""Timeseries encoding to a fixed size vector representation.

Author: Evgeny Saveliev (e.s.saveliev@gmail.com)
"""

from .seq2seq_autoencoder import (
    Encoder,
    Decoder,
    Seq2Seq,
    init_hidden,
    compute_loss,
)
from .training import train_seq2seq_autoencoder, iterate_eval_set
