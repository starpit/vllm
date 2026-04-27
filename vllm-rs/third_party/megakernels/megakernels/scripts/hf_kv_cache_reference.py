import pydra
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM


class ScriptConfig(pydra.Config):
    model: str = "meta-llama/Llama-3.1-70B-Instruct"
    prompt: str = "tell me a funny joke about cookies"
    device_map: str = "auto"
    attn_implementation: str = "eager"


    def __init__(self):
        super().__init__()
        self.extra_indices = [198, 8586]

    def l70(self):
        self.model = "meta-llama/Llama-3.1-70B-Instruct"

    def l8(self):
        self.model = "meta-llama/Llama-3.1-8B-Instruct"


def main(config: ScriptConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model)

    raw_input_ids = tokenizer(config.prompt)["input_ids"]

    raw_input_ids.extend(config.extra_indices)

    input_ids = torch.tensor(raw_input_ids).unsqueeze(0)

    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        device_map=config.device_map,
        attn_implementation=config.attn_implementation,
    )

    out = model(input_ids, use_cache=True, output_hidden_states=True)

    logits = out.logits

    preds = logits.argmax(dim=-1)

    converted_ids = tokenizer.convert_ids_to_tokens(preds.squeeze(0))

    # out = model.generate(input_ids, max_new_tokens=100, do_sample=False)

    breakpoint()


if __name__ == "__main__":
    pydra.run(main)
