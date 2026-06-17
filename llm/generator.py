from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from peft import PeftModel
import torch


class LLMGenerator:
    def __init__(self, model_name):
        # DEVICE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        if self.device == "cuda":
            print("GPU:", torch.cuda.get_device_name(0))

        # TOKENIZER
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # penting untuk stabilitas generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Tokenizer loaded!")

        # BASE MODEL
        print("Loading base model...")
        if self.device == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                device_map={"": "cpu"},  # Paksa mapping penuh ke CPU RAM
                low_cpu_mem_usage=False  # Matikan offloading otomatis ke disk
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        print("Base model loaded!")
        # LOAD LoRA ADAPTER
        print("Loading LoRA adapter...")
        self.model = PeftModel.from_pretrained(
            self.model,
            "medical_lora_adapter"
        )
        print("LoRA adapter loaded!")

        self.model.eval()

    def generate(self, query, context):

        # CLEAN / LIMIT CONTEXT (penting untuk RAG)
        context = context.strip()

        # PROMPT CHAT STYLE (lebih aman untuk TinyLlama/chat models)
        prompt = f"""<|system|>
You are a medical assistant chatbot specialized in diabetes.
Answer ONLY using the provided context.
If the answer is not in the context, say: "I don't know based on the provided context."
Do not repeat instructions, context, or question.

<|context|>
{context}

<|user|>
{query}

<|assistant|>
"""

        # TOKENIZE
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)

        print("Generating response...")

        # GENERATION (lebih deterministic → anti halu)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,

                max_new_tokens=200,
                do_sample=False,          # 🔥 penting: matikan randomness
                temperature=0.0,          # stabil
                top_p=1.0,

                repetition_penalty=1.2,

                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # DECODE
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # AMBIL HANYA BAGIAN AFTER ASSISTANT
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()

        # CLEANING OUTPUT (hapus repetisi aneh)
        lines = response.split("\n")
        cleaned = []
        seen = set()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.lower() in seen:
                continue
            seen.add(line.lower())
            cleaned.append(line)

        return " ".join(cleaned)


# =========================
# TEST
# =========================
if __name__ == "__main__":

    llm = LLMGenerator("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    context_1 = (
        "Diabetes is a chronic disease that affects blood sugar regulation. "
        "Type 1 diabetes is an autoimmune disease where the body's immune "
        "system attacks insulin-producing cells. Type 2 diabetes is "
        "characterized by insulin resistance."
    )

    question = "What is type of diabetes?"

    response = llm.generate(question, context_1)

    print("\nBot:")
    print(response)