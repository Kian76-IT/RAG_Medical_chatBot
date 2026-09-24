from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel

import torch


class LLMGenerator:

    def __init__(self, model_name):

        # ====================================================
        # DEVICE
        # ====================================================

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Using device: {self.device}"
        )

        if self.device == "cuda":

            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )

        # ====================================================
        # TOKENIZER
        # ====================================================

        print(
            "Loading tokenizer..."
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
        )

        if self.tokenizer.pad_token is None:

            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        print(
            "Tokenizer loaded!"
        )

        # ====================================================
        # BASE MODEL
        # ====================================================

        print(
            "Loading base model..."
        )

        if self.device == "cpu":

            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map={"": "cpu"},
                    low_cpu_mem_usage=True
                )
            )

        else:

            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            )

        print(
            "Base model loaded!"
        )

        # ====================================================
        # LOAD LORA ADAPTER
        # ====================================================

        print(
            "Loading LoRA adapter..."
        )

        self.model = (
            PeftModel.from_pretrained(
                self.model,
                "medical_lora_adapter"
            )
        )

        print(
            "LoRA adapter loaded!"
        )

        self.model.eval()

    # ========================================================
    # ORIGINAL GENERATE METHOD
    # ========================================================

    def generate(
        self,
        query,
        context
    ):

        context = context.strip()

        prompt = f"""<|system|>
You are a medical assistant chatbot specialized in diabetes.

Answer ONLY using the provided context.

Rules:
- Do not invent medical information.
- Do not use information outside the context.
- Answer the question directly.
- Do not repeat the context.
- Do not repeat the question.
- Do not add greetings.
- Do not add motivational statements.
- Do not add unnecessary conclusions.
- If the answer is not in the context, say:
"I don't know based on the provided context."

<|context|>
{context}

<|user|>
{query}

<|assistant|>
"""

        return self.generate_from_prompt(
            prompt
        )

    # ========================================================
    # LANGCHAIN GENERATION METHOD
    # ========================================================

    def generate_from_prompt(
        self,
        prompt
    ):

        # ----------------------------------------------------
        # TOKENIZE
        # ----------------------------------------------------

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(
            self.model.device
        )

        print(
            "Generating response..."
        )

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = self.model.generate(

                **inputs,

                max_new_tokens=100,

                do_sample=False,

                repetition_penalty=1.2,

                eos_token_id=(
                    self.tokenizer.eos_token_id
                ),

                pad_token_id=(
                    self.tokenizer.eos_token_id
                )
            )

        # ----------------------------------------------------
        # ONLY TAKE NEW TOKENS
        # ----------------------------------------------------

        input_length = (
            inputs["input_ids"].shape[1]
        )

        generated_tokens = (
            outputs[0][input_length:]
        )

        response = (
            self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            .strip()
        )

        # ----------------------------------------------------
        # CLEAN OUTPUT
        # ----------------------------------------------------

        lines = response.split("\n")

        cleaned = []

        seen = set()

        for line in lines:

            line = line.strip()

            if not line:
                continue

            normalized = line.lower()

            if normalized in seen:
                continue

            seen.add(
                normalized
            )

            cleaned.append(
                line
            )

        response = " ".join(
            cleaned
        ).strip()

        return response


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    llm = LLMGenerator(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    context = (
        "Diabetes is a chronic disease "
        "that affects blood sugar regulation. "
        "Type 1 diabetes is an autoimmune "
        "disease where the body's immune "
        "system attacks insulin-producing cells. "
        "Type 2 diabetes is characterized "
        "by insulin resistance."
    )

    question = (
        "What is type 1 diabetes?"
    )

    response = llm.generate(
        question,
        context
    )

    print(
        "\nBot:"
    )

    print(
        response
    )