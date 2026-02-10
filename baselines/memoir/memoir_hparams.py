from dataclasses import dataclass
from typing import List, Optional

from util.hparams import HyperParams


@dataclass
class MEMOIRHyperParams(HyperParams):
    # Model identification (used for caching/artifacts; evaluation still passes --model_name separately)
    model_name: str
    device: int

    # Editing hyperparameters
    edit_lr: float
    n_iter: int
    objective_optimization: str
    inner_params: List[str]

    # MEMOIR memory hyperparameters
    top_k: int = 4096
    irr_threshold: float = 0.4
    prompt_feature_agg: str = "mean_decentered"

    # Relative to NAS DATA_DIR unless absolute
    dir_background_features: str = "memoir/background_features/gpt_j_background_features.pt"

    # Context template generation (list of [max_new_tokens, n_gen])
    context_template_length_params: Optional[List[List[int]]] = None

    # Debug/logging
    verbose: bool = False

