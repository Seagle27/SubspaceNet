from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from omegaconf import OmegaConf


@dataclass
class SimulationCommands:
    create_data: bool
    train_model: bool
    evaluate_mode: bool
    save_model: bool
    save_dataset: bool
    save_to_file: bool = False
    plot_results: bool = False
    save_plots: bool = False


@dataclass
class SystemModelParams:
    N: int              # Number of Antennas
    M: int              # Number of targets
    T: int              # Snapshots
    snr: float          # in dB
    field_type: str     # ['Far', 'Near']
    signal_nature: str  # ['coherent', 'non-coherent']
    array_form: str     # ['ula', 'mra-4', 'mra-5', 'mra-6', 'mra-7', 'mra-8']
    signal_type: str = "NarrowBand"
    eta: float = 0.0
    bias: float = 0.0
    sv_noise_var: float = 0.0
    freq_values: list = field(default_factory=lambda: [0, 500])
    antenna_pattern: bool = False

@dataclass
class TrainingParams:
    samples_size: int
    train_test_ratio: float = 0.1
    epochs: int = 150
    batch_size: int = 256
    optimizer: str = "Adam"
    learning_rate: float = 0.001
    weight_decay: float = 1e-9
    step_size: int = 50
    gamma: float = 0.5
    scheduler: str = "StepLR"
    training_objective: str = "angle"
    criterion: str = "rmspe"
    balance_factor: float = 1.0
    true_doa_train: Optional[list] = None
    true_range_train: Optional[list] = None
    true_doa_test: Optional[list] = None
    true_range_test: Optional[list] = None

@dataclass
class EvaluationParams:
    criterion: str = "rmspe"
    balance_factor: float = 1.0
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    augmented_methods: list = field(default_factory=list)
    subspace_methods: list = field(default_factory=list)

@dataclass
class ModelConfig:
    model_type: str
    model_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SimulationConfig:
    system_model: SystemModelParams
    model: ModelConfig
    training: TrainingParams
    evaluation: EvaluationParams
    commands: SimulationCommands
    scenario: Optional[Dict[str, list]] = field(default_factory=dict)


def load_simulation_config(path: str) -> SimulationConfig:
    base = OmegaConf.structured(SimulationConfig)
    yaml_cfg = OmegaConf.load(path)
    cfg = OmegaConf.merge(base, yaml_cfg)
    return OmegaConf.to_object(cfg)  # Convert to regular nested dataclasses