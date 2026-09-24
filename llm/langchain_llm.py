from typing import Any, List, Optional

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun


class TinyLlamaLoRALLM(LLM):
    """
    Thin LangChain LLM wrapper around your fine-tuned TinyLlama + LoRA
    generator (llm/generator.py -> LLMGenerator).

    This is what lets "TinyLlama + LoRA Generator" from your diagram sit as
    the last step of a LangChain LCEL chain:

        prompt_template | TinyLlamaLoRALLM(generator=llm) | StrOutputParser()

    It does NOT reimplement generation logic -- it just forwards the fully
    rendered prompt to LLMGenerator.generate_from_prompt().
    """

    generator: object  # the LLMGenerator instance

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "tinyllama-lora"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self.generator.generate_from_prompt(prompt)