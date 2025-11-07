"""Code generator for recursive self-improvement.

This module generates code samples using the trained model and provides
prompts for code generation tasks.
"""

import logging
import random
from typing import Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


class CodeGenerator:
    """Generates code samples using the trained model.

    This class handles code generation with various prompts and sampling
    strategies for recursive learning.

    Attributes:
        model: The language model.
        tokenizer: The tokenizer.
        device: Device to run generation on.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: torch.device,
    ):
        """Initialize code generator.

        Args:
            model: Trained language model.
            tokenizer: Tokenizer instance.
            device: Device to run on.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        logger.info(f"CodeGenerator initialized on device: {device}")

    def generate_from_prompt(
        self,
        prompt: str,
        max_length: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
        num_return_sequences: int = 1,
    ) -> List[str]:
        """Generate code from a prompt.

        Args:
            prompt: Input prompt for generation.
            max_length: Maximum length to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.
            num_return_sequences: Number of sequences to generate.

        Returns:
            List of generated code strings.
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_codes = []
        for ids in generated_ids:
            code = self.tokenizer.decode(ids.tolist(), skip_special_tokens=True)
            # Remove the prompt from the generated code
            if code.startswith(prompt):
                code = code[len(prompt):].strip()
            generated_codes.append(code)

        return generated_codes

    def generate_batch(
        self,
        prompts: List[str],
        max_length: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> List[str]:
        """Generate code from a batch of prompts.

        Args:
            prompts: List of input prompts.
            max_length: Maximum length to generate.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.

        Returns:
            List of generated code strings.
        """
        generated_codes = []
        for prompt in prompts:
            codes = self.generate_from_prompt(
                prompt=prompt,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=1,
            )
            generated_codes.extend(codes)

        return generated_codes


class PromptGenerator:
    """Generates prompts for code generation.

    This class creates diverse prompts to encourage the model to generate
    varied and useful code samples.
    """

    # Template prompts for different types of code generation
    FUNCTION_TEMPLATES = [
        "def {func_name}({params}):\n    \"\"\"",
        "def {func_name}({params}):\n    # TODO:",
        "# Function to {description}\ndef {func_name}({params}):",
    ]

    CLASS_TEMPLATES = [
        "class {class_name}:\n    \"\"\"",
        "class {class_name}({parent}):\n    def __init__(self,",
        "# {description}\nclass {class_name}:",
    ]

    ALGORITHM_TEMPLATES = [
        "# Implement {algorithm}\ndef",
        "# Solution for: {problem}\ndef solve(",
        "def {algorithm}_algorithm(",
    ]

    # Common function names
    FUNCTION_NAMES = [
        "process_data",
        "calculate_result",
        "validate_input",
        "transform",
        "filter_items",
        "sort_data",
        "find_max",
        "compute_sum",
        "parse_input",
        "format_output",
        "search",
        "update",
        "delete",
        "create",
    ]

    # Common class names
    CLASS_NAMES = [
        "DataProcessor",
        "Calculator",
        "Validator",
        "Parser",
        "Manager",
        "Handler",
        "Controller",
        "Service",
        "Repository",
        "Model",
    ]

    # Common algorithms
    ALGORITHMS = [
        "binary_search",
        "quicksort",
        "mergesort",
        "fibonacci",
        "factorial",
        "gcd",
        "prime_check",
        "palindrome_check",
        "reverse_list",
        "find_duplicates",
    ]

    # Common descriptions
    DESCRIPTIONS = [
        "process user input",
        "validate data format",
        "calculate statistics",
        "sort items by priority",
        "filter invalid entries",
        "transform data structure",
        "handle errors gracefully",
        "optimize performance",
        "cache results",
        "manage resources",
    ]

    def __init__(self, seed: Optional[int] = None):
        """Initialize prompt generator.

        Args:
            seed: Random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)

    def generate_function_prompt(self) -> str:
        """Generate a prompt for function generation.

        Returns:
            Prompt string.
        """
        template = random.choice(self.FUNCTION_TEMPLATES)
        func_name = random.choice(self.FUNCTION_NAMES)
        params = self._generate_params()
        description = random.choice(self.DESCRIPTIONS)

        return template.format(
            func_name=func_name,
            params=params,
            description=description,
        )

    def generate_class_prompt(self) -> str:
        """Generate a prompt for class generation.

        Returns:
            Prompt string.
        """
        template = random.choice(self.CLASS_TEMPLATES)
        class_name = random.choice(self.CLASS_NAMES)
        parent = random.choice(["object", "ABC", "BaseModel"])
        description = random.choice(self.DESCRIPTIONS)

        return template.format(
            class_name=class_name,
            parent=parent,
            description=description,
        )

    def generate_algorithm_prompt(self) -> str:
        """Generate a prompt for algorithm implementation.

        Returns:
            Prompt string.
        """
        template = random.choice(self.ALGORITHM_TEMPLATES)
        algorithm = random.choice(self.ALGORITHMS)
        problem = random.choice(self.DESCRIPTIONS)

        return template.format(
            algorithm=algorithm,
            problem=problem,
        )

    def generate_docstring_prompt(self) -> str:
        """Generate a prompt for code with docstring.

        Returns:
            Prompt string.
        """
        prompts = [
            'def calculate(x, y):\n    """',
            'class DataHandler:\n    """',
            'def process_items(items: List[int]) -> List[int]:\n    """',
        ]
        return random.choice(prompts)

    def generate_comment_prompt(self) -> str:
        """Generate a prompt starting with a comment.

        Returns:
            Prompt string.
        """
        comments = [
            "# TODO: Implement",
            "# Fix:",
            "# Optimize:",
            "# Add error handling for",
            "# Refactor:",
        ]
        return random.choice(comments)

    def generate_random_prompt(self) -> str:
        """Generate a random prompt of any type.

        Returns:
            Prompt string.
        """
        generators = [
            self.generate_function_prompt,
            self.generate_class_prompt,
            self.generate_algorithm_prompt,
            self.generate_docstring_prompt,
            self.generate_comment_prompt,
        ]

        generator = random.choice(generators)
        return generator()

    def generate_prompts(self, n: int) -> List[str]:
        """Generate multiple prompts.

        Args:
            n: Number of prompts to generate.

        Returns:
            List of prompt strings.
        """
        return [self.generate_random_prompt() for _ in range(n)]

    def _generate_params(self) -> str:
        """Generate function parameters.

        Returns:
            Parameter string.
        """
        param_names = ["x", "y", "data", "items", "value", "key", "index"]
        num_params = random.randint(0, 3)

        if num_params == 0:
            return ""

        params = random.sample(param_names, min(num_params, len(param_names)))
        return ", ".join(params)


class CodeGenerationTask:
    """Manages code generation tasks for recursive learning.

    This class orchestrates the generation of code samples using both
    the model and prompt generator.
    """

    def __init__(
        self,
        code_generator: CodeGenerator,
        prompt_generator: PromptGenerator,
    ):
        """Initialize code generation task.

        Args:
            code_generator: CodeGenerator instance.
            prompt_generator: PromptGenerator instance.
        """
        self.code_generator = code_generator
        self.prompt_generator = prompt_generator

        logger.info("CodeGenerationTask initialized")

    def generate_samples(
        self,
        num_samples: int,
        max_length: int = 256,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50,
    ) -> List[Dict[str, str]]:
        """Generate multiple code samples.

        Args:
            num_samples: Number of samples to generate.
            max_length: Maximum length per sample.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling parameter.

        Returns:
            List of dictionaries with 'prompt' and 'code' keys.
        """
        logger.info(f"Generating {num_samples} code samples...")

        samples = []
        prompts = self.prompt_generator.generate_prompts(num_samples)

        for prompt in prompts:
            try:
                codes = self.code_generator.generate_from_prompt(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=1,
                )

                for code in codes:
                    samples.append({
                        "prompt": prompt,
                        "code": code,
                    })

            except Exception as e:
                logger.warning(f"Failed to generate from prompt '{prompt}': {e}")
                continue

        logger.info(f"Generated {len(samples)} samples successfully")
        return samples
