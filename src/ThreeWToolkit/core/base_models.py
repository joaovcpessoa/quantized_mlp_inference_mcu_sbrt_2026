from abc import ABC

from pydantic import BaseModel, Field, field_validator, ValidationInfo

from ..core.enums import ModelTypeEnum


class ModelsConfig(BaseModel):
    model_type: ModelTypeEnum | str = Field(..., description="Type of model to use.")
    random_seed: int | None = Field(42, description="Random seed for reproducibility.")

    @field_validator("model_type")
    @classmethod
    def check_model_type(
        cls: type["ModelsConfig"],
        value: ModelTypeEnum | str | None,
        info: ValidationInfo,
    ) -> ModelTypeEnum | str:
        """Validate that model_type is supported.

        Args:
            cls (ModelsConfig): The class reference.
            value (ModelTypeEnum | str | None): The model type to validate.
            info (ValidationInfo): Validation info.

        Returns:
            ModelTypeEnum | str: Validated model type.

        Raises:
            ValueError: If model_type is missing.
            NotImplementedError: If model_type is not supported.
        """
        valid_types = {e for e in ModelTypeEnum}
        valid_strs = {e.value for e in ModelTypeEnum}
        if value is None:
            raise ValueError("model_type is required.")
        if value not in valid_types and value not in valid_strs:
            raise NotImplementedError(f"`model_type` {value} not implemented yet.")
        return value


class BaseModels(ABC):
    def __init__(self, config: ModelsConfig):
        """
        Base model class constructor.

        Args:
            config (ModelsConfig): Configuration object with model parameters.
        """
        super().__init__()
        self.config = config
