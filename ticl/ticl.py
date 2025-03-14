import random
import json
import logging
from typing import List, Dict, Tuple, Optional, Union, Any
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplanationResponse(BaseModel):
    """Defines the schema for the explanation response."""
    explanation: str
    is_consistent: str


class GeneratedOutput(BaseModel):
    """Represents a generated output and its explanation."""
    output: str
    explanation: str


class TICLSample(BaseModel):
    """Represents a single TICL training sample."""
    task: str
    reference_output: str
    generated_outputs: List[GeneratedOutput] = []


# Helper functions moved outside the TICLSample class
def add_negative_sample(sample: TICLSample, output: str, explanation: str) -> None:
    """
    Adds a negative sample and its explanation to the sample.

    Args:
        sample: The TICLSample to update
        output: The output text to add as a negative sample
        explanation: The explanation for why the output is a negative sample
    """
    sample.generated_outputs.append(
        GeneratedOutput(output=output, explanation=explanation)
    )


class TICL:
    """Implementation of Trial-Error-Explain In-Context Learning (TICL)."""

    def __init__(self, llm_provider, iterations: int = 3, examples_per_prompt: int = 2, capacity: int = 5):
        """
        Initialize the TICL algorithm.

        Args:
            llm_provider: The LLM provider to use for generation
            iterations: Number of TICL iterations to perform
            examples_per_prompt: Number of examples to include in each ICL prompt
            capacity: Maximum number of negative samples per example
        """
        self.llm_provider = llm_provider
        self.iterations = iterations
        self.examples_per_prompt = examples_per_prompt
        self.capacity = capacity
        self.dataset = []
        self.discarded_samples = {}  # Store discarded samples by task

    def load_dataset(self, dataset: List[Dict[str, str]]):
        """
        Load the initial training dataset.

        Args:
            dataset: List of dictionaries containing tasks and reference outputs
        """
        self.dataset = [
            TICLSample(task=item["task"],
                       reference_output=item["reference_output"])
            for item in dataset
        ]
        logger.info(f"Loaded {len(self.dataset)} samples into TICL dataset")

    def _format_icl_prompt(self, target_task: str, examples: List[TICLSample]) -> str:
        """
        Format the ICL prompt according to Figure 6 in the paper.

        Args:
            target_task: The task for which to generate output
            examples: List of TICLSample objects to use as examples

        Returns:
            Formatted ICL prompt
        """
        prompt_parts = [
            "You are a stylistically consistent writer. Below are examples that exemplify your writing style."
        ]

        for i, example in enumerate(examples, 1):
            # Add the example task
            prompt_parts.append(f"# Writing Task Example {i}")
            prompt_parts.append(f"{example.task}")

            # Add the reference output
            prompt_parts.append(f"## Your Writing {i}")
            prompt_parts.append(f"{example.reference_output}")

            # Add the negative samples and explanations
            for j, neg_sample in enumerate(example.generated_outputs, 1):
                prompt_parts.append(
                    f"## Stylistically Inconsistent Writing {i}-{j}")
                prompt_parts.append(f"{neg_sample.output}")
                prompt_parts.append(f"### Inconsistent stylistic elements in 'Stylistically Inconsistent Writing {i}-{j}' that "
                                    f"should be corrected for it to become more consistent with 'Your Writing {i}':")
                prompt_parts.append(f"{neg_sample.explanation}")

        # Add the target task
        prompt_parts.extend([
            "** Task to complete **",
            "Now complete the following writing task with a style and format consistent with 'Your Writing' examples "
            "and also avoiding the stylistic inconsistencies found in the 'Stylistically Inconsistent Writing' examples.",
            "Be consistent in terms of (1) length, (2) format, (3) paragraph structure, (4) sentence structure, "
            "(5) punctuation, (6) syntax, (7) voice, and (8) diction of your writing when completing the task.",
            f"Task: {target_task}",
            "Directly provide your response in the following format:",
            "'''",
            "<your writing>",
            "'''"
        ])

        return "\n".join(prompt_parts)

    def _format_explanation_prompt(self, task: str, reference_text: str, generated_text: str) -> str:
        """
        Format the explanation prompt according to Figure 4 in the paper.

        Args:
            task: The writing task
            reference_text: The reference output (author's writing)
            generated_text: The generated output to analyze

        Returns:
            Formatted explanation prompt
        """
        prompt = f"""You are an editor.
- Your task is to analyze whether the candidate writing is stylistically consistent
with the author's writing(s) and if not, highlight elements of the author's style
that are not observed in the candidate writing.
- Consider similarity with regards to the (1) length, (2) format, (3) paragraph
structure, (4) sentence structure, (5) punctuation, (6) syntax, (7) voice, and (8)
diction of the author's writing, but NOT the content it covers.
- Use the minimum words possible in your analysis while providing specific examples
of how the observed inconsistencies must be edited to become stylistically
consistent with the author's writing.
- If the candidate writing is stylistically consistent with the author's writing,
respond with "yes" in the "is_consistent" field. Otherwise, respond with "no".
# Task
{task}
# Author's writing
{reference_text}
# Candidate writing to edit
{generated_text}
Respond only with JSON with the following format:
{{
"explanation": "<style analysis and suggested edits>",
"is_consistent": "<yes/no>"
}}"""
        return prompt

    def _generate_output(self, task: str, examples: List[TICLSample], print_prompt: bool = False) -> str:
        """
        Generate an output for the given task using the ICL prompt.

        Args:
            task: The task for which to generate output
            examples: List of TICLSample objects to use as examples
            print_prompt: Whether to print the full prompt sent to the LLM

        Returns:
            Generated output
        """
        prompt = self._format_icl_prompt(task, examples)

        messages = [
            {"role": "system", "content": "You are a stylistically consistent writer that follows instructions carefully."},
            {"role": "user", "content": prompt}
        ]

        if print_prompt:
            print("\n=== TICL Generation Prompt ===")
            print(f"System: {messages[0]['content']}")
            print(f"User Prompt:\n{messages[1]['content']}")
            print("===========================\n")

        response_text = self.llm_provider.generate(messages=messages)

        # Extract the text between triple quotes if present
        if "'''" in response_text:
            parts = response_text.split("'''")
            if len(parts) >= 3:
                return parts[1].strip()

        return response_text.strip()

    def _generate_explanation(self, task: str, reference_text: str, generated_text: str) -> Tuple[str, bool]:
        """
        Generate an explanation for why the generated text is inconsistent with the reference.

        Args:
            task: The writing task
            reference_text: The reference output (author's writing)
            generated_text: The generated output to analyze

        Returns:
            Tuple of (explanation, is_negative_sample)
        """
        prompt = self._format_explanation_prompt(
            task, reference_text, generated_text)

        messages = [
            {"role": "system", "content": "You are an expert editor analyzing writing style consistency."},
            {"role": "user", "content": prompt}
        ]

        try:
            # Use a simple response format dict instead of a pydantic model
            response_format = {"type": "json_object"}

            response_text = self.llm_provider.generate(
                messages=messages,
                response_format=response_format
            )

            # Process the response based on its type
            if isinstance(response_text, dict):
                # Response is already a dictionary
                explanation_data = response_text
            elif isinstance(response_text, str):
                # Response is a JSON string, parse it
                try:
                    explanation_data = json.loads(response_text)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse explanation response as JSON: {response_text[:100]}...")
                    return "Failed to parse explanation", True
            else:
                logger.warning(
                    f"Unexpected response type: {type(response_text)}")
                return f"Unexpected response type: {type(response_text)}", True

            # Extract explanation and is_consistent fields
            explanation = explanation_data.get(
                "explanation", "No explanation provided")
            is_consistent = explanation_data.get(
                "is_consistent", "no").lower() == "yes"

            # If it's consistent, it's not a negative sample
            is_negative_sample = not is_consistent

            return explanation, is_negative_sample

        except Exception as e:
            logger.warning(f"Error generating explanation: {e}")
            # Default to treating it as a negative sample with a generic explanation
            return f"Error generating explanation: {str(e)}", True

    def run(self, target_index: Optional[int] = None) -> List[TICLSample]:
        """
        Run the TICL algorithm on the dataset.

        Args:
            target_index: Optional index to run TICL on a specific sample
                          If None, runs on all samples

        Returns:
            The augmented dataset
        """
        if target_index is not None:
            # Run TICL on a specific sample
            indices = [target_index]
        else:
            # Run TICL on all samples
            indices = list(range(len(self.dataset)))

        # Main TICL loop
        for iteration in range(self.iterations):
            logger.info(
                f"Starting TICL iteration {iteration+1}/{self.iterations}")

            for idx in indices:
                sample = self.dataset[idx]
                logger.info(f"Processing sample {idx+1}/{len(indices)}")

                # Sample ICL examples excluding the current sample
                available_examples = [
                    s for i, s in enumerate(self.dataset) if i != idx]
                if len(available_examples) > self.examples_per_prompt:
                    icl_examples = random.sample(
                        available_examples, self.examples_per_prompt)
                else:
                    icl_examples = available_examples

                # Generate output for the current sample
                generated_output = self._generate_output(
                    sample.task, icl_examples)

                # Generate explanation and check if it's a negative sample
                explanation, is_negative_sample = self._generate_explanation(
                    sample.task, sample.reference_output, generated_output
                )

                # If it's a negative sample, add it to the dataset
                if is_negative_sample:
                    logger.info(f"Adding negative sample for sample {idx+1}")
                    # Check capacity before adding
                    if len(sample.generated_outputs) >= self.capacity:
                        # Store the discarded sample in the summary
                        discarded = sample.generated_outputs.pop(0)
                        if sample.task not in self.discarded_samples:
                            self.discarded_samples[sample.task] = []
                        self.discarded_samples[sample.task].append({
                            "output": discarded.output[:100] + "..." if len(discarded.output) > 100 else discarded.output,
                            "explanation": discarded.explanation[:100] + "..." if len(discarded.explanation) > 100 else discarded.explanation
                        })
                        logger.info(
                            f"Removed oldest negative sample to maintain capacity of {self.capacity}")

                    # Add the new negative sample
                    add_negative_sample(sample, generated_output, explanation)
                else:
                    logger.info(
                        f"Generated output is stylistically consistent - not adding as negative sample")

        return self.dataset

    def get_final_output(self, task: str, print_prompt: bool = False) -> str:
        """
        Generate the final output for a new task using the augmented dataset.

        Args:
            task: The task to generate output for
            print_prompt: Whether to print the full prompt sent to the LLM

        Returns:
            Generated output
        """
        # Sample examples from the augmented dataset
        if len(self.dataset) > self.examples_per_prompt:
            examples = random.sample(self.dataset, self.examples_per_prompt)
        else:
            examples = self.dataset

        return self._generate_output(task, examples, print_prompt)
