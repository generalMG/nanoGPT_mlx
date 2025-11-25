"""
Legacy configuration parser (DEPRECATED).

This module is deprecated and kept only for backward compatibility.
Please use command-line arguments with train.py instead:

    python train.py --n_layer=6 --n_head=6 --n_embd=384

For new projects, use the TrainingConfig dataclass in train.py.

WARNING: This module previously used exec() which is a security risk.
The current implementation uses safe parsing methods.
"""

import json
import logging
import sys
import warnings
from ast import literal_eval
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Issue deprecation warning on import
warnings.warn(
    "configurator.py is deprecated. Use command-line arguments with train.py instead. "
    "Example: python train.py --n_layer=6 --n_head=6",
    DeprecationWarning,
    stacklevel=2,
)


def parse_config_file(config_path: str) -> Dict[str, Any]:
    """
    Parse a JSON configuration file.

    Args:
        config_path: Path to a JSON config file.

    Returns:
        Dictionary of configuration values.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    with open(config_path, "r") as f:
        return json.load(f)


def parse_key_value_arg(arg: str) -> tuple[str, Any]:
    """
    Parse a --key=value command-line argument safely.

    Args:
        arg: Argument string in format '--key=value'.

    Returns:
        Tuple of (key, parsed_value).

    Raises:
        ValueError: If the argument format is invalid.
    """
    if not arg.startswith("--") or "=" not in arg:
        raise ValueError(
            f"Invalid argument format: '{arg}'. Expected '--key=value'"
        )

    key, val = arg[2:].split("=", 1)

    # Try to parse as Python literal (int, float, bool, list, etc.)
    try:
        parsed_value = literal_eval(val)
    except (SyntaxError, ValueError):
        # If literal_eval fails, treat as string
        parsed_value = val

    return key, parsed_value


def apply_config_overrides(
    base_config: Dict[str, Any],
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply configuration overrides to a base configuration.

    Args:
        base_config: Base configuration dictionary.
        overrides: Override values to apply.

    Returns:
        Updated configuration dictionary.

    Raises:
        ValueError: If an override key doesn't exist in base config.
    """
    result = base_config.copy()

    for key, value in overrides.items():
        if key not in result:
            raise ValueError(
                f"Unknown configuration key: '{key}'. "
                f"Valid keys are: {list(result.keys())}"
            )

        # Type check
        expected_type = type(result[key])
        if result[key] is not None and not isinstance(value, expected_type):
            # Try to convert
            try:
                value = expected_type(value)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Type mismatch for '{key}': expected {expected_type.__name__}, "
                    f"got {type(value).__name__}"
                ) from e

        result[key] = value
        logger.info(f"Override: {key} = {value}")

    return result


def parse_args_to_config(
    args: list[str],
    base_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Parse command-line arguments and apply to base configuration.

    This is a safe replacement for the old exec()-based configuration system.

    Args:
        args: List of command-line arguments.
        base_config: Base configuration dictionary.

    Returns:
        Updated configuration dictionary.

    Example:
        >>> base = {"n_layer": 12, "n_head": 12, "learning_rate": 6e-4}
        >>> parse_args_to_config(["--n_layer=6", "--n_head=6"], base)
        {'n_layer': 6, 'n_head': 6, 'learning_rate': 6e-4}
    """
    overrides = {}

    for arg in args:
        if "=" not in arg:
            # Assume it's a config file path
            if arg.endswith(".json"):
                file_config = parse_config_file(arg)
                overrides.update(file_config)
            else:
                logger.warning(
                    f"Skipping argument '{arg}': not a --key=value or .json file"
                )
            continue

        if arg.startswith("--"):
            key, value = parse_key_value_arg(arg)
            overrides[key] = value

    return apply_config_overrides(base_config, overrides)


# For backward compatibility, provide a simple interface
def configure(globals_dict: Dict[str, Any]) -> None:
    """
    Configure globals from command-line arguments (DEPRECATED).

    This function provides backward compatibility but is deprecated.
    Use parse_args_to_config() or command-line arguments to train.py instead.

    Args:
        globals_dict: Dictionary of global variables to update.
    """
    warnings.warn(
        "configure() is deprecated. Use train.py command-line arguments instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    overrides = {}
    for arg in sys.argv[1:]:
        if "=" not in arg:
            if arg.endswith(".json"):
                overrides.update(parse_config_file(arg))
            continue

        if arg.startswith("--"):
            try:
                key, value = parse_key_value_arg(arg)
                if key in globals_dict:
                    overrides[key] = value
                else:
                    logger.warning(f"Unknown config key: {key}")
            except ValueError as e:
                logger.error(str(e))

    globals_dict.update(overrides)
