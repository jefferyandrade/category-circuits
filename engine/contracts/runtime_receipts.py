"""The shared projection for elapsed-time fields in runtime receipts."""
from collections.abc import Mapping, MutableMapping
from dataclasses import fields, is_dataclass

HASH_EXCLUDES_WALL_TIME = True
SETTING_FIELD = "hash_excludes_wall_time"
_TIME_DERIVED_FIELDS = frozenset((
    "phase_accounting_fraction", "producer_wait_fraction",
    "predicted_to_realized_cost_ratio", "utility_per_device_second",
))


def runtime_receipt_projection(value):
    """Drop reported durations and their rates; leave work and outcomes bound."""
    if is_dataclass(value) and not isinstance(value, type):
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if isinstance(value, Mapping):
        return {
            key: runtime_receipt_projection(item)
            for key, item in value.items()
            if not (str(key).endswith("_seconds") or str(key) == "seconds"
                    or str(key).endswith("_bytes_per_second")
                    or key in _TIME_DERIVED_FIELDS)
        }
    if isinstance(value, (tuple, list)):
        return [runtime_receipt_projection(item) for item in value]
    return value


def recorded_runtime_hash_payload(value):
    """Verify a receipt using its recorded choice, including after a flip."""
    if isinstance(value, Mapping) and value.get(SETTING_FIELD) is True:
        return runtime_receipt_projection(value)
    return value


def runtime_receipt_hash(payload, *, prefix="", hash_function=None):
    """Record the wall-time choice on the receipt and hash by it."""
    if not isinstance(payload, MutableMapping):
        raise TypeError("runtime receipts must be mutable mappings when hashed")
    payload.setdefault(SETTING_FIELD, HASH_EXCLUDES_WALL_TIME)
    if type(payload[SETTING_FIELD]) is not bool:
        raise ValueError("the runtime hash timing choice must be a boolean")
    if hash_function is None:
        from mechanistic_systems.contracts.content_address import content_hash
        hash_function = content_hash
    projected = recorded_runtime_hash_payload(payload)
    if prefix:
        return hash_function(projected, prefix=prefix)
    return hash_function(projected)
