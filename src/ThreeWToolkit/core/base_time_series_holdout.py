from abc import ABC
from pydantic import BaseModel, ConfigDict
import pandas as pd


class TimeSeriesHoldoutConfig(BaseModel):
    """
    Configuration model for time series holdout splitting.

    Attributes:
        test_size (float | None): Proportion or count of test samples. Must be <= 1 if float.
        train_size (float | None): Proportion or count of train samples. Must be <= 1 if float.
        random_state (int | None): Seed for reproducibility.
        shuffle (bool): Whether to shuffle data before splitting. Defaults to False.
        stratify (pd.Series | pd.DataFrame | None): Labels for stratification. Must be set only if shuffle is True.
    """

    test_size: float | None = None
    train_size: float | None = None
    random_state: int | None = None
    shuffle: bool = False
    stratify: pd.Series | pd.DataFrame | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class BaseTimeSeriesHoldout(ABC):
    def __init__(self, config: TimeSeriesHoldoutConfig):
        """
        Abstract base class for time series holdout logic.

        Args:
            config (TimeSeriesHoldoutConfig): Configuration for the holdout split.
        """
        self.config = config
