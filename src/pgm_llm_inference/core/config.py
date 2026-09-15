from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceConfig(BaseSettings):
    # Logging
    verbose: bool = False
    log_level: str = "INFO"
    log_file_name: str = "logs/score_test.jsonl"

    # Numerical stability
    epsilon: float = 1e-10

    # Elimination ordering
    default_ordering_heuristic: str = "min_degree" # "min_degree" or "max_degree" or "topological" or "reverse_topological" or "central" or "random"
        
    show_input_data: bool = True
    show_llm_prompt: bool = False
    show_llm_output: bool = False
    mock: bool = False
    # LLM settings
    llm_timeout: float = 2400.0
    llm_max_retries: int = 3

    # OpenAI API settings (also works with compatible proxies)
    openai_api_key: str | None = None
    openai_base_url: str | None = "https://openrouter.ai/api/v1"
    openai_model: str = "deepseek/deepseek-v4-flash"
    # openai_model: str = "anthropic/claaaauuuuudeeee-sooooonneeeeet-4.6"
    # openai_model: str = "~deepseek/deepseek-v4-flash-latest"
    # openai_model: str = "qwen3-coder:30b"
    openai_temperature: float = 0.0
    openai_use_json_response_format: bool = True

    # local_url: str ="http://localhost:11434/v1/chat/completions"
    # local_model: str = "qwen3:32b-ctx"

    local_url: str ="http://localhost:11434/v1/chat/completions"
    local_model: str ="qwen3-coder:30b"
    # local_model: str ="deepseek/deepseek-r1-0528-qwen3-8b"
    # local_model: str ="meta-llama-3.1-8b-instruct"
    # local_model: str ="qwen2.5-7b-instruct-1m"
    
    mode: str = "mpe"

    model_config = SettingsConfigDict(
        env_prefix="PGM_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
