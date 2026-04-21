"""FlowRefiner (FM-Hybrid) configuration."""

from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class BaseConfig:
    """Shared data/domain settings."""
    INPUT_TIMESTEPS: int = 5
    OUTPUT_TIMESTEPS: int = 5
    INDICATORS: List[str] = None
    TOTAL_Z_LAYERS: int = 128

    def __post_init__(self):
        if self.INDICATORS is None:
            self.INDICATORS = ['u', 'v', 'w', 'p']


@dataclass
class FlowRefinerConfig(BaseConfig):
    """FlowRefiner hyperparameters.

    Key knobs (can be overridden from the CLI):
        REFINER_HIDDEN      base channel width (64 ≈ 50M params)
        REFINER_STEPS       K, number of refinement steps
        SIGMA_SCHEDULE      'ddpm' | 'ddpm_large' | 'linear' | 'fixed_range'
        ODE_STEPS           N Euler substeps per refinement step at inference
        NOISE_TYPE          flow-matching prior (default 'white')
        SIGMA_MAX/SIGMA_MIN for 'fixed_range' schedule (defaults 0.01/0.001)
    """
    REFINER_HIDDEN: int = 64
    REFINER_STEPS: int = 3
    REFINER_TIME_DIM: int = 128
    REFINER_CH_MULTS: Tuple[int, ...] = (1, 2, 2, 4)
    REFINER_N_BLOCKS: int = 2
    MIN_NOISE_STD: float = 4e-7

    SIGMA_SCHEDULE: str = 'ddpm'
    ODE_STEPS: int = 10
    NOISE_TYPE: str = 'white'
    NOISE_SPECTRUM_PATH: str = None
    SIGMA_MAX: float = None
    SIGMA_MIN: float = None


_CONFIGS = {
    'small': FlowRefinerConfig(REFINER_HIDDEN=48, REFINER_STEPS=2),
    'base':  FlowRefinerConfig(REFINER_HIDDEN=64, REFINER_STEPS=3),
    'large': FlowRefinerConfig(REFINER_HIDDEN=96, REFINER_STEPS=4),
}


def get_config(size: str = 'base') -> FlowRefinerConfig:
    if size not in _CONFIGS:
        raise ValueError(f"Unknown size {size!r}. Available: {list(_CONFIGS)}")
    return _CONFIGS[size]
