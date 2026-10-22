import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.generation.prompt_builder import build_context, build_prompt
from src.models import MinimalSource

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"


class AnswerGenerator:
    """
    Generate grounded answers with a local causal language model.
    """
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """Load the tokenizer and the model once, in evaluation mode.

        Args:
            model_name: Model name on the Hugging Face Hub.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )  # get tokenizer for model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        )  # get actual language model from Hugging Face
        # float32 calculation means precicsion for model.
        self.model.eval()  # mode generation/preciscion not tra

    def generate(
        self,
        question: str,
        sources: list[MinimalSource],
        max_new_tokens: int = 200,
    ) -> str:
        """Generate an answer to a question from its retrieved sources.

        Thinking mode is disabled so the answer is returned without
        reasoning blocks.

        Args:
            question: The question to answer.
            sources: Retrieved sources used to build the context.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            The generated answer text.
        """
        context = build_context(sources)
        messages = build_prompt(question=question, context=context)
        # -----------------
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # for answer, not contuni
            enable_thinking=False,
        )  # we changes the format for model understanding.

        inputs = self.tokenizer(
            input_text, return_tensors="pt"  # tensor
        )  # like tensor([[123, 456, 789]])

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy Decoding (be serio)
            )

        # output_ids is [input prompt tokens] + [generated answer tokens]
        # we need only answer
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]

        # print(inputs["input_ids"].shape[0]) n question
        # print(inputs["input_ids"].shape[0]) n toknes in question

        raw_answer = self.tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )
        return self._strip_thinking(raw_answer)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Remove a leftover <think> block from a generated text.

        Args:
            text: Raw generated text.

        Returns:
            The text after the closing </think> tag, or the stripped
            text if none.
        """
        if "</think>" in text:
            return text.split("</think>")[-1].strip()
        return text.strip()
