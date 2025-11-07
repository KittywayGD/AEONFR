"""Code evaluator for validating generated code samples.

This module provides safe execution and evaluation of generated code
using subprocess isolation and various quality metrics.
"""

import ast
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CodeEvaluator:
    """Evaluates generated code for correctness and quality.

    This class provides safe execution and analysis of code samples
    with various quality metrics.

    Attributes:
        timeout: Maximum execution time in seconds.
        use_docker: Whether to use Docker for isolation.
        docker_image: Docker image to use.
    """

    def __init__(
        self,
        timeout: int = 10,
        use_docker: bool = False,
        docker_image: str = "python:3.10-slim",
    ):
        """Initialize code evaluator.

        Args:
            timeout: Maximum execution time in seconds.
            use_docker: Whether to use Docker for isolation.
            docker_image: Docker image for execution.
        """
        self.timeout = timeout
        self.use_docker = use_docker
        self.docker_image = docker_image

        logger.info(
            f"CodeEvaluator initialized: timeout={timeout}s, "
            f"use_docker={use_docker}"
        )

    def evaluate(self, code: str) -> Dict[str, any]:
        """Evaluate a code sample.

        Args:
            code: Code string to evaluate.

        Returns:
            Dictionary containing evaluation results.
        """
        results = {
            "code": code,
            "valid_syntax": False,
            "execution_success": False,
            "execution_output": None,
            "execution_error": None,
            "quality_score": 0.0,
            "metrics": {},
        }

        # Check syntax
        syntax_valid, syntax_error = self._check_syntax(code)
        results["valid_syntax"] = syntax_valid
        if not syntax_valid:
            results["execution_error"] = syntax_error
            return results

        # Execute code
        success, output, error = self._execute_code(code)
        results["execution_success"] = success
        results["execution_output"] = output
        results["execution_error"] = error

        # Calculate quality metrics
        metrics = self._calculate_metrics(code)
        results["metrics"] = metrics

        # Calculate overall quality score
        results["quality_score"] = self._calculate_quality_score(results)

        return results

    def evaluate_batch(self, code_samples: List[str]) -> List[Dict[str, any]]:
        """Evaluate multiple code samples.

        Args:
            code_samples: List of code strings.

        Returns:
            List of evaluation result dictionaries.
        """
        results = []
        for i, code in enumerate(code_samples):
            logger.debug(f"Evaluating sample {i+1}/{len(code_samples)}")
            try:
                result = self.evaluate(code)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to evaluate sample {i}: {e}")
                results.append({
                    "code": code,
                    "valid_syntax": False,
                    "execution_success": False,
                    "execution_error": str(e),
                    "quality_score": 0.0,
                })

        return results

    def _check_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check if code has valid Python syntax.

        Args:
            code: Code string to check.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)

    def _execute_code(self, code: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Execute code safely in a subprocess.

        Args:
            code: Code string to execute.

        Returns:
            Tuple of (success, stdout, stderr).
        """
        if self.use_docker:
            return self._execute_in_docker(code)
        else:
            return self._execute_in_subprocess(code)

    def _execute_in_subprocess(
        self,
        code: str,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Execute code in a subprocess.

        Args:
            code: Code string to execute.

        Returns:
            Tuple of (success, stdout, stderr).
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
            ) as f:
                f.write(code)
                temp_file = f.name

            # Execute
            result = subprocess.run(
                ["python", temp_file],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            # Clean up
            Path(temp_file).unlink()

            success = result.returncode == 0
            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return False, None, "Execution timeout"
        except Exception as e:
            return False, None, str(e)

    def _execute_in_docker(
        self,
        code: str,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Execute code in a Docker container.

        Args:
            code: Code string to execute.

        Returns:
            Tuple of (success, stdout, stderr).
        """
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
            ) as f:
                f.write(code)
                temp_file = f.name

            # Docker command
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{temp_file}:/code.py:ro",
                "--network",
                "none",
                "--memory",
                "256m",
                "--cpus",
                "0.5",
                self.docker_image,
                "python",
                "/code.py",
            ]

            # Execute
            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            # Clean up
            Path(temp_file).unlink()

            success = result.returncode == 0
            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return False, None, "Execution timeout"
        except Exception as e:
            return False, None, str(e)

    def _calculate_metrics(self, code: str) -> Dict[str, any]:
        """Calculate code quality metrics.

        Args:
            code: Code string to analyze.

        Returns:
            Dictionary of metrics.
        """
        metrics = {}

        # Line count
        lines = code.strip().split("\n")
        metrics["num_lines"] = len(lines)
        metrics["num_non_empty_lines"] = len([l for l in lines if l.strip()])

        # Character count
        metrics["num_characters"] = len(code)

        # Check for docstrings
        metrics["has_docstring"] = '"""' in code or "'''" in code

        # Check for comments
        metrics["num_comments"] = len([l for l in lines if l.strip().startswith("#")])

        # Check for function definitions
        metrics["num_functions"] = len(re.findall(r"^\s*def\s+\w+", code, re.MULTILINE))

        # Check for class definitions
        metrics["num_classes"] = len(re.findall(r"^\s*class\s+\w+", code, re.MULTILINE))

        # Check for imports
        metrics["num_imports"] = len(re.findall(r"^\s*(?:import|from)\s+", code, re.MULTILINE))

        # Check for type hints
        metrics["has_type_hints"] = "->" in code or ": " in code

        # Try to parse AST for more metrics
        try:
            tree = ast.parse(code)
            metrics["num_ast_nodes"] = len(list(ast.walk(tree)))

            # Count different node types
            for node in ast.walk(tree):
                node_type = type(node).__name__
                metrics[f"num_{node_type}"] = metrics.get(f"num_{node_type}", 0) + 1

        except Exception:
            pass

        return metrics

    def _calculate_quality_score(self, results: Dict[str, any]) -> float:
        """Calculate overall quality score.

        Args:
            results: Evaluation results dictionary.

        Returns:
            Quality score between 0 and 1.
        """
        score = 0.0
        metrics = results["metrics"]

        # Valid syntax (required)
        if not results["valid_syntax"]:
            return 0.0

        score += 0.3  # Base score for valid syntax

        # Execution success
        if results["execution_success"]:
            score += 0.3

        # Code length (prefer reasonable length)
        num_lines = metrics.get("num_lines", 0)
        if 5 <= num_lines <= 100:
            score += 0.1

        # Has functions or classes
        if metrics.get("num_functions", 0) > 0 or metrics.get("num_classes", 0) > 0:
            score += 0.1

        # Has docstring
        if metrics.get("has_docstring", False):
            score += 0.1

        # Has comments
        if metrics.get("num_comments", 0) > 0:
            score += 0.05

        # Has type hints
        if metrics.get("has_type_hints", False):
            score += 0.05

        return min(score, 1.0)

    def filter_valid_samples(
        self,
        results: List[Dict[str, any]],
        min_quality_score: float = 0.5,
        require_execution_success: bool = False,
    ) -> List[Dict[str, any]]:
        """Filter evaluation results for valid samples.

        Args:
            results: List of evaluation results.
            min_quality_score: Minimum quality score threshold.
            require_execution_success: Whether to require successful execution.

        Returns:
            Filtered list of results.
        """
        filtered = []
        for result in results:
            # Check quality score
            if result["quality_score"] < min_quality_score:
                continue

            # Check execution success if required
            if require_execution_success and not result["execution_success"]:
                continue

            filtered.append(result)

        logger.info(
            f"Filtered {len(filtered)}/{len(results)} samples "
            f"(threshold={min_quality_score})"
        )

        return filtered


class CodeQualityChecker:
    """Additional quality checks for generated code.

    This class provides more sophisticated quality checks beyond
    basic execution.
    """

    # Patterns that indicate potential issues
    SUSPICIOUS_PATTERNS = [
        r"eval\(",
        r"exec\(",
        r"__import__",
        r"subprocess\.",
        r"os\.system",
        r"open\(.+['\"]w",  # Writing files
    ]

    # Patterns that indicate good practices
    GOOD_PATTERNS = [
        r'""".*"""',  # Docstrings
        r"def\s+\w+\([^)]*\)\s*->",  # Type hints
        r"if\s+__name__\s*==\s*['\"]__main__['\"]",  # Main guard
    ]

    @staticmethod
    def check_security(code: str) -> Tuple[bool, List[str]]:
        """Check code for potential security issues.

        Args:
            code: Code string to check.

        Returns:
            Tuple of (is_safe, list of warnings).
        """
        warnings = []

        for pattern in CodeQualityChecker.SUSPICIOUS_PATTERNS:
            if re.search(pattern, code):
                warnings.append(f"Suspicious pattern found: {pattern}")

        is_safe = len(warnings) == 0
        return is_safe, warnings

    @staticmethod
    def check_best_practices(code: str) -> Dict[str, bool]:
        """Check if code follows best practices.

        Args:
            code: Code string to check.

        Returns:
            Dictionary of best practice checks.
        """
        checks = {}

        # Check for docstrings
        checks["has_docstring"] = bool(re.search(r'""".*"""', code, re.DOTALL))

        # Check for type hints
        checks["has_type_hints"] = bool(re.search(r"->\s*\w+", code))

        # Check for main guard
        checks["has_main_guard"] = bool(
            re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", code)
        )

        # Check for comments
        checks["has_comments"] = bool(re.search(r"#.*", code))

        return checks

    @staticmethod
    def estimate_complexity(code: str) -> int:
        """Estimate code complexity using cyclomatic complexity.

        Args:
            code: Code string to analyze.

        Returns:
            Complexity estimate.
        """
        try:
            tree = ast.parse(code)
            complexity = 1  # Base complexity

            # Count decision points
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                    complexity += 1
                elif isinstance(node, ast.BoolOp):
                    complexity += len(node.values) - 1

            return complexity

        except Exception:
            return -1  # Invalid code
