from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel

import torch


class LLMGenerator:

    def __init__(self, model_name):

        # =========================
        # DEVICE
        # =========================

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"Using device: {self.device}")

        if self.device == "cuda":
            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )

        # =========================
        # TOKENIZER
        # =========================

        print("Loading tokenizer...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
        )

        print("Tokenizer loaded!")

        # =========================
        # BASE MODEL
        # =========================

        print("Loading base model...")

        self.model = AutoModelForCausalLM.from_pretrained(

            model_name,

            trust_remote_code=True,

            torch_dtype=(
                torch.float16
                if self.device == "cuda"
                else torch.float32
            ),

            device_map="auto"
        )

        print("Base model loaded!")

        # =========================
        # LOAD LoRA ADAPTER
        # =========================

        print("Loading LoRA adapter...")

        self.model = PeftModel.from_pretrained(
            self.model,
            "medical_lora_adapter"
        )

        print("LoRA adapter loaded!")

        # =========================
        # EVAL MODE
        # =========================

        self.model.eval()

    # =========================
    # GENERATE RESPONSE
    # =========================

    def generate(self, query, context):

        prompt = f"""
You are a helpful AI medical chatbot specialized in diabetes.

Rules:
- Use ONLY the provided context
- Do NOT hallucinate
- Keep answers short and clear
- If answer is not found, say:
"I don't have enough information."

Context:
{context}

Question:
{query}

Answer:
"""

        # =========================
        # TOKENIZE
        # =========================

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)

        print("Generating response...")

        # =========================
        # GENERATION
        # =========================

        with torch.no_grad():

            outputs = self.model.generate(

                **inputs,

                max_new_tokens=300,

                temperature=0.2,

                top_p=0.9,

                repetition_penalty=1.2,

                do_sample=True,

                eos_token_id=self.tokenizer.eos_token_id,

                pad_token_id=self.tokenizer.eos_token_id
            )

        # =========================
        # DECODE
        # =========================

        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # =========================
        # CLEAN RESPONSE
        # =========================

        if "Answer:" in response:

            response = response.split(
                "Answer:"
            )[-1].strip()

        # remove repeated lines
        lines = response.split("\n")

        cleaned_lines = []

        for line in lines:

            line = line.strip()

            if (
                line
                and line not in cleaned_lines
            ):
                cleaned_lines.append(line)

        response = " ".join(cleaned_lines)

        return response


# =========================
# TEST
# =========================

if __name__ == "__main__":

    llm = LLMGenerator(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )

    context = """
    Diabetes is a chronic disease that affects blood sugar regulation.
    """

    question = "What is diabetes?"

    response = llm.generate(
        question,
        context
    )

    print("\nBot:")
    print(response)