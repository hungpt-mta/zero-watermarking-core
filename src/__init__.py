"""Core zero-watermarking package (minimal repo)."""

from .zero_watermarking_core import (
    SystemParams,
    UserKeys,
    TempMatcher,
    generate_system_params_and_key,
    generate_user_keys,
    encrypt,
    decrypt_and_verify,
    correcting_image,
    generate_zero_watermark,
    authenticate,
)

__all__ = [
    "SystemParams",
    "UserKeys",
    "TempMatcher",
    "generate_system_params_and_key",
    "generate_user_keys",
    "encrypt",
    "decrypt_and_verify",
    "correcting_image",
    "generate_zero_watermark",
    "authenticate",
]
