#!/usr/bin/env python3
"""
Evaluate language model on SmallBench tasks.

SmallBench consists of 4 tasks:
- Sentiment (SST-2): Binary sentiment classification
- NLI (RTE): Natural Language Inference
- QA (BoolQ): Boolean question answering
- Paraphrase (MRPC): Paraphrase detection

Each task evaluates the model's ability to perform zero-shot or few-shot
classification using language modeling capabilities.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_from_disk
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.lm import LanguageModel, ModelConfig
from utils.checkpointing import load_checkpoint

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class SmallBenchEvaluator:
    """Evaluator for SmallBench tasks using language model scoring."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        device: str = "cuda",
        max_samples: Optional[int] = None,
    ):
        """Initialize evaluator.

        Args:
            model: Language model
            tokenizer: Tokenizer (HuggingFace or custom)
            device: Device to run on
            max_samples: Maximum samples per task (for quick testing)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_samples = max_samples

        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def score_completion(self, prompt: str, completion: str) -> float:
        """Score a completion given a prompt using log probability.

        Args:
            prompt: Input prompt
            completion: Completion to score

        Returns:
            Average log probability per token
        """
        # Tokenize prompt and completion
        full_text = prompt + completion
        full_ids = self.tokenizer.encode(full_text)
        prompt_ids = self.tokenizer.encode(prompt)

        # Convert to tensors
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)

        # Forward pass
        logits, _ = self.model(input_ids)

        # Compute log probs for completion tokens
        # logits: [1, seq_len, vocab]
        # We want to score tokens from len(prompt_ids) to end
        completion_start = len(prompt_ids) - 1  # -1 because logits are shifted
        completion_logits = logits[0, completion_start:-1, :]  # [completion_len, vocab]
        completion_targets = input_ids[0, completion_start + 1:]  # [completion_len]

        # Get log probs
        log_probs = torch.log_softmax(completion_logits, dim=-1)
        token_log_probs = log_probs[range(len(completion_targets)), completion_targets]

        # Return average log prob
        return token_log_probs.mean().item()

    def evaluate_sentiment(self, data_dir: Path) -> Dict[str, float]:
        """Evaluate on sentiment classification (SST-2).

        Args:
            data_dir: Path to SmallBench data directory

        Returns:
            Dictionary with accuracy and other metrics
        """
        logger.info("Evaluating sentiment task (SST-2)...")

        # Load dataset
        task_path = data_dir / "sentiment"
        if not task_path.exists():
            logger.warning(f"Sentiment data not found at {task_path}")
            return {"accuracy": 0.0, "num_samples": 0}

        dataset = load_from_disk(str(task_path))

        # Take subset if specified
        examples = list(dataset)
        if self.max_samples:
            examples = examples[:self.max_samples]

        correct = 0
        total = 0

        for example in tqdm(examples, desc="Sentiment"):
            sentence = example["sentence"]
            label = example["label"]  # 0: negative, 1: positive

            # Create prompt and completions
            prompt = f"Sentence: {sentence}\nSentiment:"
            pos_completion = " positive"
            neg_completion = " negative"

            # Score both completions
            pos_score = self.score_completion(prompt, pos_completion)
            neg_score = self.score_completion(prompt, neg_completion)

            # Predict based on higher score
            prediction = 1 if pos_score > neg_score else 0

            if prediction == label:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0

        logger.info(f"Sentiment: {correct}/{total} = {accuracy:.2%}")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "num_samples": len(examples),
        }

    def evaluate_nli(self, data_dir: Path) -> Dict[str, float]:
        """Evaluate on NLI (RTE).

        Args:
            data_dir: Path to SmallBench data directory

        Returns:
            Dictionary with accuracy and other metrics
        """
        logger.info("Evaluating NLI task (RTE)...")

        # Load dataset
        task_path = data_dir / "nli"
        if not task_path.exists():
            logger.warning(f"NLI data not found at {task_path}")
            return {"accuracy": 0.0, "num_samples": 0}

        dataset = load_from_disk(str(task_path))

        # Take subset if specified
        examples = list(dataset)
        if self.max_samples:
            examples = examples[:self.max_samples]

        correct = 0
        total = 0

        for example in tqdm(examples, desc="NLI"):
            premise = example["sentence1"]
            hypothesis = example["sentence2"]
            label = example["label"]  # 0: entailment, 1: not_entailment

            # Create prompt and completions
            prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nRelation:"
            entail_completion = " entailment"
            not_entail_completion = " not entailment"

            # Score both completions
            entail_score = self.score_completion(prompt, entail_completion)
            not_entail_score = self.score_completion(prompt, not_entail_completion)

            # Predict based on higher score
            prediction = 0 if entail_score > not_entail_score else 1

            if prediction == label:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0

        logger.info(f"NLI: {correct}/{total} = {accuracy:.2%}")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "num_samples": len(examples),
        }

    def evaluate_qa(self, data_dir: Path) -> Dict[str, float]:
        """Evaluate on QA (BoolQ).

        Args:
            data_dir: Path to SmallBench data directory

        Returns:
            Dictionary with accuracy and other metrics
        """
        logger.info("Evaluating QA task (BoolQ)...")

        # Load dataset
        task_path = data_dir / "qa"
        if not task_path.exists():
            logger.warning(f"QA data not found at {task_path}")
            return {"accuracy": 0.0, "num_samples": 0}

        dataset = load_from_disk(str(task_path))

        # Take subset if specified
        examples = list(dataset)
        if self.max_samples:
            examples = examples[:self.max_samples]

        correct = 0
        total = 0

        for example in tqdm(examples, desc="QA"):
            passage = example["passage"]
            question = example["question"]
            label = example["label"]  # True or False

            # Create prompt and completions
            prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
            yes_completion = " yes"
            no_completion = " no"

            # Score both completions
            yes_score = self.score_completion(prompt, yes_completion)
            no_score = self.score_completion(prompt, no_completion)

            # Predict based on higher score
            prediction = True if yes_score > no_score else False

            if prediction == label:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0

        logger.info(f"QA: {correct}/{total} = {accuracy:.2%}")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "num_samples": len(examples),
        }

    def evaluate_paraphrase(self, data_dir: Path) -> Dict[str, float]:
        """Evaluate on paraphrase detection (MRPC).

        Args:
            data_dir: Path to SmallBench data directory

        Returns:
            Dictionary with accuracy and other metrics
        """
        logger.info("Evaluating paraphrase task (MRPC)...")

        # Load dataset
        task_path = data_dir / "paraphrase"
        if not task_path.exists():
            logger.warning(f"Paraphrase data not found at {task_path}")
            return {"accuracy": 0.0, "num_samples": 0}

        dataset = load_from_disk(str(task_path))

        # Take subset if specified
        examples = list(dataset)
        if self.max_samples:
            examples = examples[:self.max_samples]

        correct = 0
        total = 0

        for example in tqdm(examples, desc="Paraphrase"):
            sentence1 = example["sentence1"]
            sentence2 = example["sentence2"]
            label = example["label"]  # 0: not paraphrase, 1: paraphrase

            # Create prompt and completions
            prompt = f"Sentence 1: {sentence1}\nSentence 2: {sentence2}\nParaphrase:"
            yes_completion = " yes"
            no_completion = " no"

            # Score both completions
            yes_score = self.score_completion(prompt, yes_completion)
            no_score = self.score_completion(prompt, no_completion)

            # Predict based on higher score
            prediction = 1 if yes_score > no_score else 0

            if prediction == label:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0

        logger.info(f"Paraphrase: {correct}/{total} = {accuracy:.2%}")

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "num_samples": len(examples),
        }

    def evaluate_all(self, data_dir: Path) -> Dict[str, Dict[str, float]]:
        """Evaluate on all SmallBench tasks.

        Args:
            data_dir: Path to SmallBench data directory

        Returns:
            Dictionary with results for all tasks
        """
        results = {}

        results["sentiment"] = self.evaluate_sentiment(data_dir)
        results["nli"] = self.evaluate_nli(data_dir)
        results["qa"] = self.evaluate_qa(data_dir)
        results["paraphrase"] = self.evaluate_paraphrase(data_dir)

        # Compute average accuracy
        accuracies = [r["accuracy"] for r in results.values() if r["accuracy"] > 0]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

        results["average"] = {"accuracy": avg_accuracy}

        logger.info(f"\n{'=' * 60}")
        logger.info("SmallBench Results:")
        logger.info(f"  Sentiment: {results['sentiment']['accuracy']:.2%}")
        logger.info(f"  NLI:       {results['nli']['accuracy']:.2%}")
        logger.info(f"  QA:        {results['qa']['accuracy']:.2%}")
        logger.info(f"  Paraphrase: {results['paraphrase']['accuracy']:.2%}")
        logger.info(f"  Average:   {avg_accuracy:.2%}")
        logger.info(f"{'=' * 60}\n")

        return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on SmallBench")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/smallbench",
        help="Path to SmallBench data directory",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="data/lm_tokenized/tokenizer.json",
        help="Path to tokenizer",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per task (for testing)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer}")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)

    # Load model from checkpoint
    logger.info(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Get model config from checkpoint
    if "model_config" in checkpoint:
        model_config = checkpoint["model_config"]
    else:
        # Infer from state dict
        logger.warning("Model config not in checkpoint, using default")
        model_config = ModelConfig()

    # Create model
    model = LanguageModel(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)

    logger.info(f"Model loaded successfully on {args.device}")

    # Create evaluator
    evaluator = SmallBenchEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        max_samples=args.max_samples,
    )

    # Run evaluation
    data_dir = Path(args.data_dir)
    results = evaluator.evaluate_all(data_dir)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
