from dataclasses import dataclass, field


MAP = "map"
MPE = "mpe"
INFERENCE_MODES = {MAP, MPE}


@dataclass
class ExperimentConfig:
    """Configuration shared by experiment runners and executable scripts."""

    dataset_name: str
    use_real_llm: bool = True
    use_local_llm: bool = True
    inference_mode: str = MPE
    context_type: str | None = None
    prompt_types: list[str] = field(default_factory=lambda: ["simple"])
    query_sizes: list[int] = field(default_factory=lambda: [1])
    n_trials: int = 5
    max_context_rows_per_call: int = 96
    max_hidden_variables: int = 300
    evidence_sampling: str = "mpe_inconsistent"

    def __post_init__(self) -> None:
        if self.inference_mode not in INFERENCE_MODES:
            raise ValueError(f"Invalid mode: {self.inference_mode}")
