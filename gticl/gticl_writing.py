import re
import logging
import enum
import sys
import traceback
from typing import Dict, List, Tuple, Optional, Any, Union, Literal
from pydantic import BaseModel, Field

from .llm_providers import LLMProvider
from .dataset_utils import load_dataset

logger = logging.getLogger(__name__)

# Pydantic models for structured LLM outputs


class CandidateRanking(BaseModel):
    """Defines the ranking and explanation for a single candidate."""
    candidate_index: int
    rank: int
    standalone_explanation: str
    comparative_explanation: str


class EvaluationResponse(BaseModel):
    """Defines the schema for ranking multiple candidates."""
    rankings: List[CandidateRanking]


class EvaluationCriteriaResponse(BaseModel):
    """Defines the schema for evaluation criteria generation."""
    evaluation_criteria: str

# Config


class GTICLConfig(BaseModel):
    """Configuration for GTICL."""
    k_candidates: int = 3  # Number of candidates to generate per input
    capacity: int = 5      # Max number of negative samples to store per task
    iterations: int = 2    # Number of overall refinement iterations
    use_rewrite_update: bool = False  # Whether to repeatedly refine rewrites
    examples_per_prompt: int = 2  # Number of examples to include in each prompt


class GTICLSample(BaseModel):
    """Represents a single GTICL training sample."""
    task: str
    reference_output: str


class GTICL:
    """
    GRPO Trial-Error-Explain In-Context Learning (GTICL)

    A framework for personalized, tuning-free style alignment using:
    - Multi-sample generation
    - LLM-based evaluation
    - Explanation + rewrite of negative samples
    - Iterative refinement with bounded storage
    """

    def __init__(self, llm_provider: LLMProvider, config: Optional[GTICLConfig] = None):
        """
        Initialize the GTICL framework.

        Args:
            llm_provider: Provider for LLM generation
            config: Configuration for GTICL framework
        """
        self.llm_provider = llm_provider
        self.config = config or GTICLConfig()
        self.negative_samples_store = {}  # task -> list of negative samples
        self.negative_samples_summary = {}  # task -> summary of discarded samples
        self.dataset = []  # List of GTICLSample objects

    def load_dataset(self, dataset_path: str = None, dataset: List[Dict[str, str]] = None):
        """
        Load the dataset for GTICL.

        Args:
            dataset_path: Path to the dataset file (optional)
            dataset: Pre-loaded dataset (optional)
        """
        if dataset:
            self.dataset = [GTICLSample(task=item["task"], reference_output=item["reference_output"])
                            for item in dataset]
        elif dataset_path:
            raw_dataset = load_dataset(dataset_path)
            self.dataset = [GTICLSample(task=item["task"], reference_output=item["reference_output"])
                            for item in raw_dataset]
        else:
            logger.warning("No dataset provided to load")

        logger.info(f"Loaded {len(self.dataset)} samples into GTICL dataset")

    def generate_candidates(self,
                            task: str,
                            training_samples: List[Tuple[str, str]] = None,
                            k_candidates: int = None) -> List[str]:
        """
        Generate k candidate outputs for a given task.

        Args:
            task: Task description
            training_samples: List of (task, reference output) pairs
            k_candidates: Number of candidates to generate

        Returns:
            List of generated candidate outputs
        """
        if k_candidates is None:
            k_candidates = self.config.k_candidates

        # Get examples from the dataset if not provided
        if not training_samples and self.dataset:
            # Use examples from dataset, limited by examples_per_prompt
            training_samples = [(sample.task, sample.reference_output)
                                for sample in self.dataset[:self.config.examples_per_prompt]]

        if not training_samples:
            training_samples = []

        # Get negative examples for the task if they exist
        task_negative_samples = self.negative_samples_store.get(task, [])

        candidates = []
        for _ in range(k_candidates):
            # Build prompt using Template P from specs
            messages = self._build_generation_prompt(
                task, training_samples, task_negative_samples)

            # Generate a candidate
            response = self.llm_provider.generate(messages)

            # Extract output from response
            candidate = self._extract_output_from_response(response)
            candidates.append(candidate)

        return candidates

    def _build_generation_prompt(self,
                                 task: str,
                                 training_samples: List[Tuple[str, str]],
                                 negative_samples: List[Dict]) -> List[Dict[str, str]]:
        """Build the prompt for generating candidates."""
        system_message = "You are a highly skilled expert completing the following task."

        # Build examples part of the prompt
        examples_text = ""

        # Include reference examples and associated negative examples if available
        for i, (example_task, example_output) in enumerate(training_samples):
            example_num = i + 1
            examples_text += f"# Writing Task Example {example_num}\n"
            examples_text += f"{example_task}\n"
            examples_text += f"## Your Writing {example_num}\n{example_output}\n\n"

            # Find negative samples that might be related to this example
            example_negative_samples = [
                ns for ns in negative_samples
                if ns.get("task") == example_task or not ns.get("task")
            ][:2]  # Limit to 2 negative samples per example

            # Add negative examples for this training example
            for j, neg_sample in enumerate(example_negative_samples):
                examples_text += f"## Stylistically Inconsistent Writing {example_num}-{j+1}\n"
                examples_text += f"{neg_sample['candidate']}\n"
                examples_text += f"### Inconsistent stylistic elements in 'Stylistically Inconsistent Writing {example_num}-{j+1}' that\n"
                examples_text += f"should be corrected for it to become more consistent with 'Your Writing {example_num}':\n"
                examples_text += f"{neg_sample['explanation']}\n\n"

        user_message = f"""
You are a stylistically consistent writer. Below are examples that exemplify your
writing style.
{examples_text}
** Task to complete **
Now complete the following writing task with a style and format consistent with
'Your Writing' examples and also avoiding the stylistic inconsistencies found in the
'Stylistically Inconsistent Writing' examples.
Be consistent in terms of (1) length, (2) format, (3) paragraph structure, (4) sentence
structure, (5) punctuation, (6) syntax, (7) voice, and (8) diction of your writing when
completing the task.
Task: {task}
Directly provide your response in the following format:
'''
<your writing>
'''
"""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

    def _extract_output_from_response(self, response: str) -> str:
        """
        Extract the output from a response that might be wrapped in backticks
        or other formatting.
        """
        # Try to extract content between backticks
        backtick_match = re.search(
            r'```(?:\w*\n)?(.*?)```', response, re.DOTALL)
        if (backtick_match):
            return backtick_match.group(1).strip()

        # If no backticks, return the raw response
        return response.strip()

    def evaluate_and_rank(self,
                          candidates: List[str],
                          reference: str,
                          task: str) -> List[Tuple[str, int, str]]:
        """
        Evaluate and rank all candidates together.

        Args:
            candidates: List of candidate outputs to evaluate
            reference: Reference output (correct example)
            task: Task description

        Returns:
            List of (candidate, rank, explanation) tuples ranked from worst to best
        """

        # Build prompt for evaluating all candidates at once
        messages = self._build_evaluation_prompt(
            candidates, reference, task)

        try:
            # Define a response format that avoids using Pydantic for direct parsing
            response_format = {"type": "json_object"}

            # Get ranking response
            response_text = self.llm_provider.generate(
                messages=messages,
                response_format=response_format
            )

            # Process the response based on its type
            if isinstance(response_text, dict):
                # Response is already a dictionary
                response_obj = response_text
            elif isinstance(response_text, str):
                # Response is a JSON string, parse it
                try:
                    import json
                    response_obj = json.loads(response_text)
                except json.JSONDecodeError:
                    logger.error(
                        f"Failed to parse response as JSON: {response_text}")
                    raise ValueError("Invalid JSON response")
            else:
                logger.error(
                    f"Unexpected response type: {type(response_text)}")
                raise ValueError(
                    f"Unexpected response type: {type(response_text)}")

            # Validate the rankings exist
            if "rankings" not in response_obj:
                logger.error(
                    f"No 'rankings' field in response: {response_obj}")
                raise ValueError("No rankings in response")

            # Process rankings
            rankings = response_obj["rankings"]

            # Validate each ranking has the required fields
            for rank in rankings:
                required_fields = ["candidate_index",
                                   "rank", "standalone_explanation"]
                missing = [
                    field for field in required_fields if field not in rank]
                if missing:
                    logger.error(f"Missing fields in ranking: {missing}")
                    logger.error(f"Ranking: {rank}")
                    raise ValueError(
                        f"Missing required fields in ranking: {missing}")

            # Map candidate indexes to their rankings and explanations
            results = []
            for ranking in rankings:
                candidate_index = ranking["candidate_index"]
                if 0 <= candidate_index < len(candidates):
                    # Add to results with the original candidate text
                    results.append(
                        (candidates[candidate_index],
                         ranking["rank"],
                         ranking["standalone_explanation"])
                    )
                else:
                    logger.warning(
                        f"Invalid candidate index: {candidate_index}")

            # Verify we have rankings for all candidates
            if len(results) != len(candidates):
                logger.error(
                    f"Not all candidates were ranked. Expected {len(candidates)} rankings but got {len(results)}")
                raise ValueError("Incomplete candidate rankings")

            # Sort by rank from worst to best (higher rank number = worse)
            results.sort(key=lambda x: -x[1])
            return results

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            try:
                print("Response type:", type(response_text))
                print("Response:", response_text)
            except UnboundLocalError:
                print("No response received from the language model")
            traceback.print_exc()

            # Return default rankings instead of exiting
            logger.info("Falling back to default rankings")
            default_results = [(candidate, i+1, "Default ranking due to evaluation error")
                               for i, candidate in enumerate(candidates)]
            return default_results

    def _build_evaluation_prompt(self,
                                 candidates: List[str],
                                 reference: str,
                                 task: str) -> List[Dict[str, str]]:
        """Build the prompt for evaluating and ranking multiple candidates."""
        system_message = "You are an expert evaluator for tasks."

        # Format all candidates with numbers
        candidates_text = ""
        for i, candidate in enumerate(candidates):
            candidates_text += f"\n\n## Candidate {i}\n{candidate}"

        user_message = f"""You are an editor.

- Your task is to analyze whether the candidate writing is stylistically consistent
with the author’s writing(s) and if not, highlight elements of the author’s style
that are not observed in the candidate writing.
- Consider similarity with regards to the (1) length, (2) format, (3) paragraph
structure, (4) sentence structure, (5) punctuation, (6) syntax, (7) voice, and (8)
diction of the author’s writing, but NOT the content it covers.
- Then rank all the candidate outputs based on how stylistically consistent with the author’s writing they are.
- Rank the candidates from 1 (best) to {len(candidates)} (worst).
- Each candidate must have a unique rank (no ties).
- For each candidate, provide TWO different explanations:
  1. A standalone_explanation that uses the minimum words possible in the analysis while providing specific examples of how the observed inconsistencies must be edited to become stylistically
consistent with the author’s writing, without referring to other candidates or using the word "candidate".
  2. A comparative_explanation that explicitly compares this candidate to others and explains its relative ranking.

# Task
{task}

# Author’s writing
{reference}

# Candidates writing to edit
{candidates_text}

### **Expected Response Format (JSON)**
```json
{{
  "rankings": [
    {{
      "candidate_index": 0,
      "rank": 2,
      "standalone_explanation": "This output meets criteria X and Y, but lacks Z...",
      "comparative_explanation": "Compared to candidate 1, this candidate is weaker in..."
    }},
    {{
      "candidate_index": 1,
      "rank": 1,
      "standalone_explanation": "This output excellently addresses all criteria...",
      "comparative_explanation": "This is the strongest candidate because..."
    }},
    ...
  ]
}}
```
Ensure each candidate_index corresponds to the candidate number above (0 for Candidate 0, etc.).
"""

        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ]

    def select_negative_samples(self,
                                ranked_candidates: List[Tuple[str, int, str]],
                                max_samples: int = 1) -> List[Dict]:
        """
        Select worst candidates as negative samples.

        Args:
            ranked_candidates: List of (candidate, rank, explanation) tuples ranked from worst to best
            max_samples: Maximum number of samples to select

        Returns:
            List of dictionaries with candidate and explanation keys
        """
        negative_samples = []

        # Select up to max_samples from the worst candidates
        # ranked_candidates is already sorted with worst candidates first
        for i, (candidate, rank, explanation) in enumerate(ranked_candidates):
            # Take the max_samples worst candidates
            if i >= max_samples:
                break

            negative_samples.append({
                "candidate": candidate,
                "explanation": explanation
            })

        return negative_samples

    def explain_and_rewrite(self,
                            reference: str,
                            negative_sample: Dict,
                            task: str) -> Dict:
        """
        Generate an explanation and optionally rewrite a negative sample.

        Args:
            reference: Reference example output
            negative_sample: Dictionary containing candidate and explanation
            task: Task description

        Returns:
            Dictionary with original content plus rewrite (if use_rewrite_update is True)
        """
        result = negative_sample.copy()

        # If use_rewrite_update is True and we don't already have a rewrite,
        # generate a rewrite
        if self.config.use_rewrite_update and "rewrite" not in result:
            # Use the generation prompt to create a rewrite
            training_samples = [(task, reference)]
            neg_samples = [negative_sample]

            messages = self._build_generation_prompt(
                "Rewrite the negative sample to avoid the identified issues",
                training_samples,
                neg_samples
            )

            response = self.llm_provider.generate(messages)
            rewrite = self._extract_output_from_response(response)
            result["rewrite"] = rewrite

        return result

    def update_negative_samples_store(self,
                                      task: str,
                                      negative_samples: List[Dict],
                                      capacity: Optional[int] = None) -> None:
        """
        Update the negative samples store for a task.

        Args:
            task: Task identifier
            negative_samples: List of negative sample dictionaries
            capacity: Maximum capacity for the store (default: self.config.capacity)
        """
        if capacity is None:
            capacity = self.config.capacity

        # Initialize stores if they don't exist
        if task not in self.negative_samples_store:
            self.negative_samples_store[task] = []
            self.negative_samples_summary[task] = ""

        for sample in negative_samples:
            # Add task to the sample
            sample["task"] = task

            # Add to the store
            self.negative_samples_store[task].append(sample)

            # If we exceed capacity, remove the oldest sample and update summary
            if len(self.negative_samples_store[task]) > capacity:
                oldest = self.negative_samples_store[task].pop(0)
                summary = f"Discarded sample with issues: {oldest['explanation'][:50]}...\n"
                self.negative_samples_summary[task] += summary

    def run_alignment(self,
                      target_index: Optional[int] = None) -> Dict:
        """
        Run the alignment process on dataset examples.

        Args:
            target_index: Optional index to run GTICL on a specific sample
                          If None, runs on all samples

        Returns:
            Dictionary with negative samples store and summary
        """
        if not self.dataset:
            logger.warning("No dataset available for alignment")
            return {
                "negative_samples_store": self.negative_samples_store,
                "negative_samples_summary": self.negative_samples_summary
            }

        # Determine which examples to process
        if target_index is not None:
            # Run GTICL on a specific sample
            indices = [target_index]
        else:
            # Run GTICL on all samples
            indices = list(range(len(self.dataset)))

        # For each iteration
        for iteration in range(self.config.iterations):
            logger.info(
                f"Starting iteration {iteration+1}/{self.config.iterations}")

            for idx in indices:
                sample = self.dataset[idx]
                logger.info(f"Processing sample {idx+1}/{len(indices)}")

                # Get task and reference from the current sample
                task = sample.task
                reference_output = sample.reference_output

                # Create training samples excluding the current sample
                available_examples = [(self.dataset[i].task, self.dataset[i].reference_output)
                                      for i in range(len(self.dataset)) if i != idx]

                # Use up to examples_per_prompt examples
                training_samples = available_examples[:
                                                      self.config.examples_per_prompt]

                # Generate candidates
                candidates = self.generate_candidates(
                    task,
                    training_samples,
                    self.config.k_candidates
                )
                for j, candidate in enumerate(candidates):
                    logger.info(
                        f"Candidate {j+1}/{len(candidates)}: {candidate}")

                # Evaluate and rank
                ranked_candidates = self.evaluate_and_rank(
                    candidates,
                    reference_output,
                    task
                )
                for j, (candidate, rank, explanation) in enumerate(ranked_candidates):
                    logger.info(
                        f"Candidate {j+1}/{len(ranked_candidates)}: Explanation:{explanation} \n (Rank: {rank})")

                # Select negative samples
                worst_samples = self.select_negative_samples(ranked_candidates)
                for j, sample in enumerate(worst_samples):
                    logger.info(
                        f"Negative Sample {j+1}/{len(worst_samples)}: {sample['candidate']}")

                # Process each negative sample
                processed_samples = []
                for sample in worst_samples:
                    # Explain and rewrite
                    processed = self.explain_and_rewrite(
                        reference_output, sample, task)
                    processed_samples.append(processed)

                # Update store
                self.update_negative_samples_store(
                    task,
                    processed_samples,
                    self.config.capacity
                )

            # Refine rewrites for all tasks if needed
            if self.config.use_rewrite_update and iteration < self.config.iterations - 1:
                for task in self.negative_samples_store.keys():
                    # Find a reference for this task in the dataset
                    ref_samples = [
                        (sample.task, sample.reference_output)
                        for sample in self.dataset if sample.task == task
                    ]
                    if ref_samples:
                        self._refine_rewrites(task, ref_samples)

        return {
            "negative_samples_store": self.negative_samples_store,
            "negative_samples_summary": self.negative_samples_summary
        }

    def _refine_rewrites(self,
                         task: str,
                         training_samples: List[Tuple[str, str]]) -> None:
        """
        Refine rewrites in the negative samples store.

        Args:
            task: Task identifier
            training_samples: List of (task, reference) pairs
        """
        if task not in self.negative_samples_store:
            return

        # Get reference from the first training sample
        if not training_samples:
            return

        _, reference = training_samples[0]

        # Refine each sample
        for i, sample in enumerate(self.negative_samples_store[task]):
            # Only refine if there's already a rewrite
            if "rewrite" in sample:
                refined = self.explain_and_rewrite(reference, sample, task)
                self.negative_samples_store[task][i] = refined

    def generate_with_alignment(self,
                                task: str,
                                training_samples: List[Tuple[str, str]] = None,
                                print_prompt: bool = False) -> Dict:
        """
        Generate a final output using the alignment data.

        Args:
            task: Task description
            training_samples: List of (task, reference) pairs
            print_prompt: Whether to print the full prompt sent to the LLM

        Returns:
            Dictionary with the generated output and info about the process
        """
        # If no training samples are provided, get them from the dataset
        if not training_samples and self.dataset:
            training_samples = [(sample.task, sample.reference_output)
                                for sample in self.dataset[:self.config.examples_per_prompt]]

        if not training_samples:
            training_samples = []

        # Get negative examples for the task if they exist
        task_negative_samples = self.negative_samples_store.get(task, [])
        task_summary = self.negative_samples_summary.get(task, "")

        # Use a single generation with all our knowledge
        messages = self._build_generation_prompt(
            task, training_samples, task_negative_samples)

        if print_prompt:
            print("\n=== GTICL Generation Prompt ===")
            print(f"System: {messages[0]['content']}")
            print(f"User Prompt:\n{messages[1]['content']}")
            print("===========================\n")

        response = self.llm_provider.generate(messages)

        output = self._extract_output_from_response(response)

        return {
            "output": output,
            "negative_samples_used": len(task_negative_samples),
            "has_summary": bool(task_summary)
        }
