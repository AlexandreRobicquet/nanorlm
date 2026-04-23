DEFAULT_ENDPOINTS_PATH = "./configs/endpoints.toml"


def validate_eval_config(raw: dict) -> None:
    raw_endpoint_id = raw.get("endpoint_id")
    if raw_endpoint_id is not None and not str(DEFAULT_ENDPOINTS_PATH).endswith(".toml"):
        raise ValueError(
            "'endpoint_id' is only supported with TOML endpoint registries. "
            "Set endpoints_path to an endpoints.toml file."
        )


def is_valid_eval_results_path(path: str) -> bool:
    return path.endswith("results.jsonl") or path.endswith("metadata.json")
