import torch
from transformers import (
    AutoTokenizer,
    BatchEncoding,
    PreTrainedTokenizerBase,
    Qwen3_5ForConditionalGeneration,
)
from transformers.cli.serving.utils import get_response_template

from data.tool_schemas import (
    calculate_expression,
    create_note,
    get_weather,
    watch_tiktok,
)


def load_model(
    model_id: str = 'Qwen/Qwen3.5-0.8B',
) -> tuple[Qwen3_5ForConditionalGeneration, PreTrainedTokenizerBase]:
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_id,
        dtype=model_dtype,
        device_map='auto',
    ).eval()
    return model, tokenizer


def encode_prompt(
    model: Qwen3_5ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
) -> BatchEncoding:
    return tokenizer.apply_chat_template(
        messages,
        tools=[
            get_weather,
            create_note,
            watch_tiktok,
            calculate_expression,
        ],
        enable_thinking=False,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors='pt',
    ).to(model.device)


@torch.inference_mode()
def run_prompt(
    model: Qwen3_5ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
) -> dict:
    model_inputs = encode_prompt(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
    )
    generated_token_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,
        do_sample=False,
    )
    return tokenizer.parse_response(
        response=generated_token_ids[0, model_inputs['input_ids'].shape[1] :],
        schema=get_response_template(
            processor=tokenizer,
            model=model,
        ),
        prefix=model_inputs['input_ids'][0],
    )


@torch.inference_mode()
def run_prompt_with_steering(
    model: Qwen3_5ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict],
    steering_vector: torch.Tensor,
    layer_index: int = 12,
    steering_strength: float = 1.0,
) -> dict:

    def _add_steering_vector(
        _module: torch.nn.Module,
        _module_inputs: tuple[torch.Tensor, ...],
        layer_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        layer_hidden_states = layer_hidden_states.clone()
        layer_hidden_states[:, -1, :] += steering_strength * steering_vector.to(
            device=layer_hidden_states.device,
            dtype=layer_hidden_states.dtype,
        )
        return layer_hidden_states

    transformer_layer = model.model.language_model.layers[layer_index]
    hook_handle = transformer_layer.register_forward_hook(_add_steering_vector)
    try:
        return run_prompt(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
        )
    finally:
        hook_handle.remove()
