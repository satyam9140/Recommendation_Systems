from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectConfig:
    data_dir: Path = Path("data/sample")
    output_dir: Path = Path("outputs/demo")
    min_user_ratings: int = 3
    min_movie_ratings: int = 3
    test_ratio: float = 0.2
    relevant_threshold: float = 3.5
    top_k: int = 10
    random_state: int = 42


DEFAULT_CONFIG = ProjectConfig()
