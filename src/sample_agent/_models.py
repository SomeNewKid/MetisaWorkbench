"""Models used by the sample agent."""

from pydantic import BaseModel, ConfigDict, Field


class ProductDetails(BaseModel):
    """Describe a product."""

    model_config = ConfigDict(extra="forbid", strict=True)

    product_code: str = Field(min_length=1)
    product_name: str = Field(min_length=1)


class PhoneNumber(BaseModel):
    """Provide a phone number."""

    model_config = ConfigDict(extra="forbid", strict=True)

    phone_number: str = Field(min_length=1, max_length=10)
