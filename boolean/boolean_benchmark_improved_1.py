import sys
import json
import argparse
import os
import re
import yaml
import random
import numpy as np
import ast
import sympy as sp
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Callable
from datetime import datetime
from textwrap import dedent
from collections import Counter
from scipy import stats
import traceback

# Add parent directory to path if needed
sys.path.append(str(Path(__file__).parent))

from modules.llm_interface import LLMInterface, OpenRouterLLM, OpenAILLM, AnthropicLLM
from boolean_dataset import BooleanExpression, BooleanObservation


class BooleanBenchmarkImproved:
    def __init__(self, complete_dataset_path: str):
        """
        Initialize benchmark with a complete dataset.
        
        Args:
            complete_dataset_path: Path to the complete Boolean dataset JSON file
        """
        with open(complete_dataset_path, 'r') as f:
            self.complete_dataset = json.load(f)
        
        # Extract metadata
        self.mechanistic_opts = self.complete_dataset['metadata']['mechanistic_opts']
        self.metadata = self.complete_dataset['metadata']
        self.variables = self.metadata['variables']
        self.operators = set(self.metadata['operators'])
        self.max_depth = self.metadata['max_depth']
        
        # Flatten all datasets into a single list for sampling
        self.all_observation_sets = []
        for n_obs, datasets in self.complete_dataset['datasets_by_n_observations'].items():
            self.all_observation_sets.extend(datasets)
        
        print(f"Loaded complete dataset with {len(self.all_observation_sets)} observation sets")
        print(f"Variables: {', '.join(self.variables)}")
        print(f"Operators: {', '.join(self.operators)}")
        print(f"Max depth: {self.max_depth}")
    
    def sample_observation_sets(self, n_samples: int, seed: Optional[int] = None) -> List[Dict]:
        """
        Sample n observation sets from the complete dataset.
        
        Args:
            n_samples: Number of observation sets to sample
            seed: Random seed for reproducibility
        
        Returns:
            List of sampled observation sets
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Sample without replacement
        n_available = len(self.all_observation_sets)
        n_to_sample = min(n_samples, n_available)
        
        if n_to_sample < n_samples:
            print(f"Warning: Requested {n_samples} samples but only {n_available} available")
        
        sampled = random.sample(self.all_observation_sets, n_to_sample)
        return sampled

    def create_prompt(self, observations: List[BooleanObservation], 
                     prior_hypotheses: List[str]) -> str:
        # Build obs_block, prior_block, operators_str ...
                # Format observations
        obs_lines = []
        for obs in observations:
            obs_lines.append(obs.to_string())
        obs_block = "\n".join(obs_lines)
        
        # Format prior hypotheses
        if prior_hypotheses:
            prior_block = "\n".join([f"Expression: {h}" for h in prior_hypotheses])
        else:
            prior_block = "None"
        
        # Build operator description
        op_descriptions = []
        if 'AND' in self.operators: op_descriptions.append("AND (conjunction)")
        if 'OR'  in self.operators: op_descriptions.append("OR (disjunction)")
        if 'NOT' in self.operators: op_descriptions.append("NOT (negation)")
        if 'XOR' in self.operators: op_descriptions.append("XOR (exclusive or)")
        if 'NOR' in self.operators: op_descriptions.append("NOR (NOT (x OR y))")
        
        operators_str = ", ".join(op_descriptions)

        # Pull mechanistic opts from the loaded dataset
        mech = self.complete_dataset.get('metadata', {}).get('mechanistic_opts', {})
        comm = mech.get('apply_commutativity', True)
        idem = mech.get('apply_idempotence_and_or', True)
        flat = mech.get('flatten_associativity', True)

        eq_rules = []
        if comm:
            eq_rules.append(
                "- Commutativity: reorderings are the SAME (e.g., x AND y = y AND x)."
            )
        if idem:
            eq_rules.append(
                "- Idempotence: duplicates of the same input under AND/OR collapse (e.g., x AND x = x; x OR x = x)."
            )
        if flat:
            eq_rules.append(
                "- Associativity flattening: cascades of the SAME operator are the SAME regardless of parentheses "
                "(e.g., (x AND y) AND x = x AND (y AND x) = x AND x AND y)."
            )
        else:
            eq_rules.append(
                "- No associativity flattening: different parenthesizations of the SAME operator are DIFFERENT "
                "(e.g., (x AND y) AND x ≠ x AND (y AND x))."
            )

        eq_rules.append(
            "- No distributivity, absorption, or De Morgan normalization: mixing operators keeps expressions DIFFERENT "
            "(e.g., (x AND y) OR z ≠ x AND (y OR z))."
        )

        rules_text = "\n        ".join(eq_rules)

        prompt = f"""You are given partial observations of a Boolean function with variables: {', '.join(self.variables)}

            Allowed operators: {operators_str}

            Observations (input -> output):
            {obs_block}

            Prior expressions generated (avoid repeating any expression 
            that is equivalent under the rules below):
            {prior_block}

            Task: Generate a single Boolean expression that is consistent with ALL observations.

            Requirements:
            1. Use ONLY the variables: {', '.join(self.variables)}
            2. Use ONLY these operators: {', '.join(self.operators)}
            3. The expression must match all given observations
            4. Structural uniqueness is judged by these rules:
            {rules_text}
            5. Expression depth should be at most {self.max_depth} levels of nesting
            6. Do not use boolean constants True or False anywhere in the expression.

            Output format: 
            - Return ONLY the Boolean expression on a single line
            - Use plain text format (no LaTeX, no markdown, no special formatting)
            - Use uppercase for operators. 
            - Use lowercase for variables: x, y
            - Use parentheses for grouping when needed
            - Start your response with "Expression: " followed by the expression
            
            Examples of correct format:
            Expression: x AND y
            Expression: (x OR y) AND NOT x
            Expression: NOT (x AND y)
            Expression: NOR(x, y)
            
            DO NOT use formats like:
            - \\(x \\land y\\)  (LaTeX)
            - `x AND y`  (markdown)
            - x ∧ y  (mathematical symbols)
            """
        return prompt
    
    def parse_llm_response(self, response: str) -> Optional[str]:
        """Parse LLM response to extract Boolean expression."""
        if not isinstance(response, str):
            return None
        
        response = response.strip()
        
        # Clean common formatting issues
        def clean_expression(expr: str) -> str:
            # Remove LaTeX delimiters
            expr = re.sub(r'\\[()\[\]]', '', expr)
            expr = re.sub(r'\$+', '', expr)
            
            # Remove markdown backticks
            expr = expr.strip('`')
            
            # Convert LaTeX/math symbols to text
            expr = expr.replace('\\land', 'AND')
            expr = expr.replace('\\lor', 'OR')
            expr = expr.replace('\\lnot', 'NOT')
            expr = expr.replace('\\neg', 'NOT')
            expr = expr.replace('\\wedge', 'AND')
            expr = expr.replace('\\vee', 'OR')
            expr = expr.replace('∧', 'AND')
            expr = expr.replace('∨', 'OR')
            expr = expr.replace('¬', 'NOT')
            expr = expr.replace('⊕', 'XOR')
            
            # Remove \text{} wrappers
            expr = re.sub(r'\\text\{([^}]+)\}', r'\1', expr)
            
            # Clean up extra spaces and symbols
            expr = re.sub(r'\*+$', '', expr)  # Remove trailing asterisks
            expr = expr.strip('"\'').rstrip('.,;')
            
            return expr.strip()
        
        # Look for "Expression:" line
        lines = response.split('\n')
        for line in lines:
            if 'Expression:' in line or 'expression:' in line:
                parts = re.split(r'[Ee]xpression:', line)
                if len(parts) >= 2:
                    expr = clean_expression(parts[1])
                    if expr:
                        return expr
        
        # Fallback: try first non-empty line with operators
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                cleaned = clean_expression(line)
                if any(op in cleaned.upper() for op in ['AND','OR','NOT','XOR','NOR']):
                    return cleaned
                if all(c.isalnum() or c in '() ' for c in cleaned):
                    return cleaned
        
        return None

    def get_expression_mechanistic_key(self, expr_str: str) -> Optional[Tuple]:
        try:
            expr = BooleanExpression(expr_str, self.variables, self.operators)
            return expr.mechanistic_key(**self.mechanistic_opts)
        except:
            return None
    
    def _in_space(self, expr_str: str) -> bool:
        """Check if expression meets all space constraints (variables, operators, depth, no constants)."""
        try:
            expr = BooleanExpression(expr_str, self.variables, self.operators)
            
            if expr.sympy_expr is None:
                return False
            
            # Check variables
            allowed_symbols = {sp.symbols(v) for v in self.variables}
            if not expr.sympy_expr.free_symbols.issubset(allowed_symbols):
                return False
            
            # Check operators
            if not self._check_operators_in_ast(expr.sympy_expr, self.operators):
                return False
            
            # Check no constants
            for node in sp.preorder_traversal(expr.sympy_expr):
                if isinstance(node, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
                    return False
            
            # Check depth
            if self._get_ast_depth(expr.sympy_expr) > self.max_depth:
                return False
            
            return True
        except:
            return False
    
    def _get_ast_depth(self, sympy_expr) -> int:
        """Calculate the true AST depth of an expression."""
        if sympy_expr is None:
            return 0
        
        # Leaf nodes (variables, constants) have depth 0
        if isinstance(sympy_expr, sp.Symbol):
            return 0
        if isinstance(sympy_expr, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
            return 0
        
        # For operators, depth is 1 + max depth of children
        if hasattr(sympy_expr, 'args') and len(sympy_expr.args) > 0:
            child_depths = [self._get_ast_depth(arg) for arg in sympy_expr.args]
            return 1 + max(child_depths)
        
        return 0
    
    def _check_operators_in_ast(self, sympy_expr, allowed_ops: Set[str]) -> bool:
        if sympy_expr is None:
            return False
        stack = [sympy_expr]
        while stack:
            node = stack.pop()

            # NOR pattern: Not(Or(...))
            if isinstance(node, sp.Not) and isinstance(node.args[0], sp.Or):
                if 'NOR' not in allowed_ops:
                    return False
                # treat as primitive; do NOT traverse inside
                continue

            # primitive checks
            if isinstance(node, sp.Not) and 'NOT' not in allowed_ops:
                return False
            elif isinstance(node, sp.And) and 'AND' not in allowed_ops:
                return False
            elif isinstance(node, sp.Or) and 'OR' not in allowed_ops:
                return False
            elif isinstance(node, sp.Xor) and 'XOR' not in allowed_ops:
                return False
            elif isinstance(node, sp.Implies) and 'IMPLIES' not in allowed_ops:
                return False
            elif isinstance(node, sp.Equivalent) and 'EQUIV' not in allowed_ops:
                return False

            if hasattr(node, 'args'):
                stack.extend(node.args)
        return True
    
    def validate_expression(self, expr_str: str, observations: List[BooleanObservation]) -> Tuple[bool, Optional[Dict]]:
        """Validate if expression is consistent with observations and meets constraints."""
        try:
            # Now create expression and validate
            expr = BooleanExpression(expr_str, self.variables, self.operators)
            
            # Check if expression uses only allowed variables
            if expr.sympy_expr is not None:
                allowed_symbols = {sp.symbols(v) for v in self.variables}
                free_symbols = expr.sympy_expr.free_symbols
                if not free_symbols.issubset(allowed_symbols):
                    return False, None
                
                # Check operators using AST traversal
                if not self._check_operators_in_ast(expr.sympy_expr, self.operators):
                    return False, None
                
                # Disallow constants (True/False) to maintain consistent space
                for node in sp.preorder_traversal(expr.sympy_expr):
                    if isinstance(node, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
                        return False, None
            
            # Check expression depth using AST
            if expr.sympy_expr is not None:
                ast_depth = self._get_ast_depth(expr.sympy_expr)
                if ast_depth > self.max_depth:
                    return False, None
            
            # Check consistency with observations
            for obs in observations:
                if expr.evaluate(obs.inputs) != obs.output:
                    return False, None
            
            return True, expr.truth_table
        except:
            return False, None
    
    def _classify_error(self, error_message: str) -> str:
        """Classify the type of error from the error message with more detail."""
        if "codec can't encode" in error_message.lower() or "UnicodeEncodeError" in error_message:
            return "request_encoding_error"
        elif "Expecting value" in error_message:
            # Extract line and column info if available
            match = re.search(r'line (\d+) column (\d+)', error_message)
            if match:
                return f"json_parse_error (line {match.group(1)}, col {match.group(2)})"
            return "json_parse_error"
        elif "Rate limit" in error_message.lower():
            return "rate_limit"
        elif "timeout" in error_message.lower():
            return "timeout"
        elif "401" in error_message or "unauthorized" in error_message.lower():
            return "auth_error"
        elif "403" in error_message or "forbidden" in error_message.lower():
            return "forbidden_error"
        elif "404" in error_message:
            return "not_found_error"
        elif "429" in error_message:
            return "rate_limit_429"
        elif "500" in error_message or "internal server error" in error_message.lower():
            return "server_error_500"
        elif "502" in error_message or "bad gateway" in error_message.lower():
            return "bad_gateway_502"
        elif "503" in error_message or "service unavailable" in error_message.lower():
            return "service_unavailable_503"
        elif "connection" in error_message.lower():
            return "connection_error"
        elif "JSONDecodeError" in error_message:
            return "json_decode_error"
        else:
            # Try to extract HTTP status code
            match = re.search(r'\b(\d{3})\b', error_message)
            if match:
                return f"http_error_{match.group(1)}"
            return "unknown_error"
    
    def get_expression_depth(self, expr_str: str) -> int:
        """Calculate the depth of a Boolean expression."""
        # Simple heuristic: count maximum nesting level of parentheses
        max_depth = 0
        current_depth = 0
        
        for char in expr_str:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        
        # Also check for operators without parentheses
        if max_depth == 0:
            # Count operators to estimate depth
            expr_upper = expr_str.upper()
            if any(op in expr_upper for op in ['AND','OR','NOT','XOR','NOR']):
                max_depth = 1
        
        return max_depth
    
    def _candidate_diagnostic(
        self,
        expression: str,
        observations: List[BooleanObservation]
    ) -> Optional[str]:
        """Explain candidate failures using only visible constraints and observations."""
        try:
            candidate = BooleanExpression(expression, self.variables, self.operators)
        except Exception:
            return "The expression could not be parsed."

        if candidate.sympy_expr is None:
            return "The expression could not be parsed."

        allowed_symbols = {sp.symbols(variable) for variable in self.variables}
        unexpected_symbols = candidate.sympy_expr.free_symbols - allowed_symbols
        if unexpected_symbols:
            names = ", ".join(sorted(str(symbol) for symbol in unexpected_symbols))
            return f"The expression uses variables outside the allowed set: {names}."

        if not self._check_operators_in_ast(candidate.sympy_expr, self.operators):
            return "The expression uses an operator outside the allowed operator set."

        if any(
            isinstance(node, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse))
            for node in sp.preorder_traversal(candidate.sympy_expr)
        ):
            return "Boolean constants are not allowed in this hypothesis space."

        if self._get_ast_depth(candidate.sympy_expr) > self.max_depth:
            return f"The expression exceeds the maximum depth of {self.max_depth}."

        try:
            for index, observation in enumerate(observations, start=1):
                actual = candidate.evaluate(observation.inputs)
                if actual != observation.output:
                    return (
                        f"The expression does not match observation {index} "
                        f"({observation.to_string()})."
                    )
        except Exception:
            return "The expression could not be evaluated against the observations."

        return None

    def _create_feedback_prompt(
        self,
        observations: List[BooleanObservation],
        valid_unique_archive: List[str],
        previous_feedback: Optional[str]
    ) -> str:
        prompt = self.create_prompt(observations, valid_unique_archive)
        feedback = previous_feedback or (
            "There is no previous candidate to diagnose. Follow the observations "
            "and all stated constraints."
        )
        return (
            f"{prompt}\n\n"
            "Feedback on the previous candidate (based only on the observations "
            "and constraints shown above):\n"
            f"{feedback}\n"
            "Provide one new candidate. Only archived expressions above have "
            "already passed validation; do not repeat their structures."
        )

    def evaluate_single_observation_set(
        self,
        llm: LLMInterface,
        observation_set: Dict,
        n_queries: int = 10,
        verbose: bool = True,
        max_retries: int = 1,
        resume_state: Optional[Dict] = None,
        progress_callback: Optional[Callable[[Dict], None]] = None
    ) -> Dict:
        observations = [
            BooleanObservation(obs['inputs'], obs['output'])
            for obs in observation_set['observations']
        ]

        state = resume_state or {}
        all_hypotheses = state.get('all_hypotheses', [])
        valid_hypotheses = state.get('valid_hypotheses', [])
        unique_valid_expressions = state.get('unique_valid_expressions', [])
        unique_all_expressions = state.get('unique_all_expressions', [])
        parse_success_count = state.get('parse_success_count', 0)
        in_space_count = state.get('in_space_count', 0)
        total_prompt_tokens = state.get('total_prompt_tokens', 0)
        total_completion_tokens = state.get('total_completion_tokens', 0)
        total_tokens = state.get('total_tokens', 0)
        total_cost = state.get('total_cost', 0.0)
        errors = state.get('errors', [])
        error_counts = state.get('error_counts', {})
        raw_responses = state.get('raw_responses', [])
        feedback_history = state.get('feedback_history', [])
        valid_unique_archive = state.get('valid_unique_archive', [])
        previous_feedback = state.get('previous_feedback')
        query_index = state.get('query_index', 0)
        attempt = state.get('attempt', 0)
        query_error = state.get('query_error')

        unique_mechanistic_keys = {
            key for expression in unique_valid_expressions
            if (key := self.get_expression_mechanistic_key(expression)) is not None
        }
        all_unique_mechanistic_keys = {
            key for expression in unique_all_expressions
            if (key := self.get_expression_mechanistic_key(expression)) is not None
        }
        archive_keys = {
            key for expression in valid_unique_archive
            if (key := self.get_expression_mechanistic_key(expression)) is not None
        }

        def persist_progress():
            if progress_callback is None:
                return
            progress_callback({
                'observation_set_id': observation_set['observation_set_id'],
                'n_queries': n_queries,
                'query_index': query_index,
                'attempt': attempt,
                'query_error': query_error,
                'all_hypotheses': all_hypotheses,
                'valid_hypotheses': valid_hypotheses,
                'unique_valid_expressions': unique_valid_expressions,
                'unique_all_expressions': unique_all_expressions,
                'parse_success_count': parse_success_count,
                'in_space_count': in_space_count,
                'total_prompt_tokens': total_prompt_tokens,
                'total_completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens,
                'total_cost': total_cost,
                'errors': errors,
                'error_counts': error_counts,
                'raw_responses': raw_responses,
                'feedback_history': feedback_history,
                'valid_unique_archive': valid_unique_archive,
                'previous_feedback': previous_feedback
            })

        # Persist an initial cursor before the first request.
        persist_progress()

        while query_index < n_queries:
            prompt = self._create_feedback_prompt(
                observations, valid_unique_archive, previous_feedback
            )

            if hasattr(llm, 'query_with_usage'):
                result = llm.query_with_usage(prompt)
                response = result['response']
                usage = result['usage']
                total_prompt_tokens += usage['prompt_tokens']
                total_completion_tokens += usage['completion_tokens']
                total_tokens += usage['total_tokens']
                total_cost += result.get('cost', 0.0)
            else:
                response = llm.query(prompt)

            raw_responses.append({
                'query_index': query_index,
                'attempt': attempt + 1,
                'response': response
            })

            if response.startswith("Error querying"):
                query_error = {
                    'query_index': query_index,
                    'attempt': attempt + 1,
                    'error_message': response,
                    'error_type': self._classify_error(response)
                }
                error_type = query_error['error_type']
                error_counts[error_type] = error_counts.get(error_type, 0) + 1
                previous_feedback = (
                    "The previous request failed before returning a candidate. "
                    "Try again using the observations and constraints shown."
                )
                attempt += 1
                if attempt >= max_retries:
                    errors.append(query_error)
                    all_hypotheses.append(
                        f"[ERROR after {max_retries} attempts] Type: "
                        f"{query_error['error_type']} | {query_error['error_message']}"
                    )
                    query_index += 1
                    attempt = 0
                    query_error = None
                persist_progress()
                continue

            hypothesis_str = self.parse_llm_response(response)
            if hypothesis_str is None:
                previous_feedback = (
                    "The previous response did not contain a parseable Boolean "
                    "expression in the required 'Expression: ...' format."
                )
                feedback_history.append({
                    'query_index': query_index,
                    'attempt': attempt + 1,
                    'feedback': previous_feedback
                })
                attempt += 1
                if attempt >= max_retries:
                    query_index += 1
                    attempt = 0
                    query_error = None
                persist_progress()
                continue

            parse_success_count += 1
            all_hypotheses.append(hypothesis_str)
            is_in_space = self._in_space(hypothesis_str)
            if is_in_space:
                in_space_count += 1
                all_mech_key = self.get_expression_mechanistic_key(hypothesis_str)
                if all_mech_key and all_mech_key not in all_unique_mechanistic_keys:
                    all_unique_mechanistic_keys.add(all_mech_key)
                    unique_all_expressions.append(hypothesis_str)

            is_valid, _ = self.validate_expression(hypothesis_str, observations)
            diagnostic = self._candidate_diagnostic(hypothesis_str, observations)

            if is_valid:
                valid_hypotheses.append(hypothesis_str)
                mech_key = self.get_expression_mechanistic_key(hypothesis_str)
                if mech_key and mech_key not in unique_mechanistic_keys:
                    unique_mechanistic_keys.add(mech_key)
                    unique_valid_expressions.append(hypothesis_str)

            if not is_in_space:
                previous_feedback = diagnostic or (
                    "The expression violates a hypothesis-space constraint."
                )
            elif not is_valid:
                previous_feedback = diagnostic or (
                    "The expression does not satisfy all provided observations."
                )
            else:
                mech_key = self.get_expression_mechanistic_key(hypothesis_str)
                if mech_key is None:
                    previous_feedback = (
                        "The candidate passed the visible checks but could not be "
                        "assigned a structural key, so it was not archived. Try "
                        "another parseable structure."
                    )
                elif mech_key in archive_keys:
                    previous_feedback = (
                        "The candidate is valid but structurally duplicates an "
                        "already archived candidate. Try a different structure."
                    )
                else:
                    valid_unique_archive.append(hypothesis_str)
                    archive_keys.add(mech_key)
                    previous_feedback = (
                        "The candidate passed all visible checks and was added to "
                        "the valid, structurally unique archive. Try another "
                        "structurally different candidate."
                    )

            feedback_history.append({
                'query_index': query_index,
                'attempt': attempt + 1,
                'feedback': previous_feedback
            })
            query_index += 1
            attempt = 0
            query_error = None
            persist_progress()

        ground_truth_expressions = observation_set['ground_truth_expressions']
        gt_mechanistic_keys = []
        for ground_truth in ground_truth_expressions:
            key = ground_truth.get('mechanistic_key')
            if key:
                gt_mechanistic_keys.append(
                    ast.literal_eval(key) if isinstance(key, str) else key
                )

        parse_success_rate = parse_success_count / n_queries if n_queries > 0 else 0
        in_space_rate = in_space_count / n_queries if n_queries > 0 else 0
        valid_rate = len(valid_hypotheses) / n_queries if n_queries > 0 else 0
        novelty_rate = len(unique_all_expressions) / n_queries if n_queries > 0 else 0

        generated_mech_keys = set()
        for expression in unique_valid_expressions:
            key = self.get_expression_mechanistic_key(expression)
            if key is not None:
                generated_mech_keys.add(key)
        recovered_gts = {
            index for index, key in enumerate(gt_mechanistic_keys)
            if key in generated_mech_keys
        }
        recovery_rate = (
            len(recovered_gts) / len(gt_mechanistic_keys)
            if gt_mechanistic_keys else 0
        )

        return {
            'observation_set_id': observation_set['observation_set_id'],
            'n_observations': observation_set['n_observations'],
            'n_ground_truths': len(ground_truth_expressions),
            'n_queries': n_queries,
            'n_valid': len(valid_hypotheses),
            'n_unique_valid': len(unique_valid_expressions),
            'n_unique_all': len(unique_all_expressions),
            'n_recovered_gts': len(recovered_gts),
            'parse_success_rate': parse_success_rate,
            'in_space_rate': in_space_rate,
            'valid_rate': valid_rate,
            'novelty_rate': novelty_rate,
            'recovery_rate': recovery_rate,
            'all_hypotheses': all_hypotheses,
            'valid_hypotheses': valid_hypotheses,
            'unique_valid_expressions': unique_valid_expressions,
            'unique_all_expressions': unique_all_expressions,
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens
            },
            'cost': total_cost,
            'errors': errors,
            'error_summary': {
                'total_errors': len(errors),
                'error_types': error_counts
            },
            'raw_responses': raw_responses,
            'valid_unique_archive': valid_unique_archive,
            'feedback_history': feedback_history
        }

    def run_benchmark(
        self,
        llm: LLMInterface,
        n_samples: int = 10,
        n_queries_per_sample: Optional[int] = None,
        query_multiplier: float = 2.0,
        seed: Optional[int] = None,
        verbose: bool = True,
        checkpoint_dir: str = "checkpoints",
        run_id: Optional[str] = None,
        max_retries: int = 1
    ) -> Dict:
        """
        Run the benchmark with sampling.
        
        Args:
            llm: LLM interface to use
            n_samples: Number of observation sets to sample
            n_queries_per_sample: Fixed number of queries per observation set (if None, uses query_multiplier)
            query_multiplier: Multiplier for n_gt to determine queries (default 2.0 means 2x ground truths)
            seed: Random seed
            verbose: Print progress
            checkpoint_dir: Directory to save checkpoints
        
        Returns:
            Dictionary with complete benchmark results
        """
        if not run_id:
            run_id = "improved_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        elif not str(run_id).startswith("improved_"):
            run_id = f"improved_{run_id}"

        print("Method: improved (validated unique archive + verification feedback)")

        # Create checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Sanitize LLM name for filename
        safe_llm_name = llm.get_name().replace('/', '_').replace('(', '_').replace(')', '_').replace(' ', '_')
        checkpoint_file = checkpoint_path / f"checkpoint_{safe_llm_name}_{run_id}.json"
        
        print(f"\nRunning improved Boolean benchmark")
        print(f"LLM: {llm.get_name()}")
        print(f"Sampling {n_samples} observation sets")
        if n_queries_per_sample is not None:
            print(f"Queries per sample: {n_queries_per_sample} (fixed)")
        else:
            print(f"Queries per sample: {query_multiplier}x number of ground truths (adaptive)")
        print(f"Checkpoint file: {checkpoint_file}")
        print("-" * 50)
        
        # Sample observation sets
        sampled_sets = self.sample_observation_sets(n_samples, seed)
        
        # Initialize results tracking
        all_results = []
        valid_rates = []
        novelty_rates = []
        recovery_rates = []
        
        # Initialize token and cost tracking
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        
        # Initialize error tracking
        all_errors = []
        total_error_counts = {}
        current_sample = None
        
        # Load existing checkpoint if it exists
        start_idx = 0
        if checkpoint_file.exists():
            print(f"Using Recovered checkpoint file: {checkpoint_file}")
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                checkpoint_version = checkpoint_data.get('checkpoint_version')
                is_legacy_checkpoint = checkpoint_version is None
                if checkpoint_version not in (3, None):
                    raise ValueError(
                        "This checkpoint does not contain per-response progress. "
                        "Use a new --run-id to start a new improved run."
                    )
                if is_legacy_checkpoint:
                    print(
                        "Legacy checkpoint detected: completed samples will be kept; "
                        "an unfinished sample will restart once. New work will use "
                        "per-response checkpoints."
                    )
                expected_settings = {
                    'run_id': run_id,
                    'llm_name': llm.get_name(),
                    'n_samples': n_samples,
                    'n_queries_per_sample': n_queries_per_sample,
                    'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
                    'seed': seed
                }
                if not is_legacy_checkpoint:
                    expected_settings['max_retries'] = max_retries
                for setting, expected in expected_settings.items():
                    if checkpoint_data.get(setting) != expected:
                        raise ValueError(
                            f"Checkpoint settings mismatch for '{setting}': "
                            f"expected {expected!r}, found {checkpoint_data.get(setting)!r}. "
                            "Use a different --run-id or restore the original settings."
                        )

                all_results = checkpoint_data['results']
                start_idx = len(all_results)
                if start_idx > len(sampled_sets):
                    raise ValueError(
                        f"Checkpoint contains {start_idx} completed samples, "
                        f"but this run has only {len(sampled_sets)} samples."
                    )
                for idx, result in enumerate(all_results):
                    expected_id = sampled_sets[idx]['observation_set_id']
                    if result.get('observation_set_id') != expected_id:
                        raise ValueError(
                            f"Checkpoint sample {idx + 1} does not match the current "
                            "dataset sampling. Use a different --run-id."
                        )
                print(f"Resuming from checkpoint: {start_idx}/{n_samples} completed")

                # Recalculate rates from checkpoint
                for result in all_results:
                    valid_rates.append(result['valid_rate'])
                    novelty_rates.append(result['novelty_rate'])
                    recovery_rates.append(result['recovery_rate'])

                # Rebuild totals from completed samples; checkpoint totals also
                # include the partial sample, which will be restored separately.
                total_prompt_tokens = sum(
                    result.get('token_usage', {}).get('prompt_tokens', 0)
                    for result in all_results
                )
                total_completion_tokens = sum(
                    result.get('token_usage', {}).get('completion_tokens', 0)
                    for result in all_results
                )
                total_tokens = sum(
                    result.get('token_usage', {}).get('total_tokens', 0)
                    for result in all_results
                )
                total_cost = sum(result.get('cost', 0.0) for result in all_results)
                all_errors = [
                    error
                    for result in all_results
                    for error in result.get('errors', [])
                ]
                total_error_counts = {}
                for result in all_results:
                    for error_type, count in result.get(
                        'error_summary', {}
                    ).get('error_types', {}).items():
                        total_error_counts[error_type] = (
                            total_error_counts.get(error_type, 0) + count
                        )
                current_sample = checkpoint_data.get('current_sample')
                if is_legacy_checkpoint:
                    current_sample = None
                if current_sample is not None:
                    if current_sample.get('sample_index') != start_idx:
                        raise ValueError(
                            "Checkpoint in-progress sample index does not match "
                            "the completed sample count."
                        )
                    if start_idx >= len(sampled_sets):
                        raise ValueError(
                            "Checkpoint has an in-progress sample but the requested "
                            "sample count is already complete."
                        )
                    expected_id = sampled_sets[start_idx]['observation_set_id']
                    if current_sample.get('observation_set_id') != expected_id:
                        raise ValueError(
                            "Checkpoint in-progress sample does not match the "
                            "current dataset sampling."
                        )

        def save_checkpoint(in_progress_sample=None):
            token_usage = {
                'prompt_tokens': sum(
                    result.get('token_usage', {}).get('prompt_tokens', 0)
                    for result in all_results
                ),
                'completion_tokens': sum(
                    result.get('token_usage', {}).get('completion_tokens', 0)
                    for result in all_results
                ),
                'total_tokens': sum(
                    result.get('token_usage', {}).get('total_tokens', 0)
                    for result in all_results
                )
            }
            total_cost_checkpoint = sum(
                result.get('cost', 0.0) for result in all_results
            )
            error_types = {}
            for result in all_results:
                for error_type, count in result.get(
                    'error_summary', {}
                ).get('error_types', {}).items():
                    error_types[error_type] = error_types.get(error_type, 0) + count

            if in_progress_sample is not None:
                progress = in_progress_sample['progress']
                token_usage['prompt_tokens'] += progress['total_prompt_tokens']
                token_usage['completion_tokens'] += progress['total_completion_tokens']
                token_usage['total_tokens'] += progress['total_tokens']
                total_cost_checkpoint += progress['total_cost']
                for error_type, count in progress['error_counts'].items():
                    error_types[error_type] = error_types.get(error_type, 0) + count

            checkpoint_data = {
                'checkpoint_version': 3,
                'method': 'improved',
                'run_id': run_id,
                'llm_name': llm.get_name(),
                'n_samples': n_samples,
                'n_queries_per_sample': n_queries_per_sample,
                'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
                'seed': seed,
                'max_retries': max_retries,
                'timestamp': datetime.now().isoformat(),
                'results': all_results,
                'current_sample': in_progress_sample,
                'total_token_usage': token_usage,
                'total_cost': total_cost_checkpoint,
                'total_errors': sum(
                    result.get('error_summary', {}).get('total_errors', 0)
                    for result in all_results
                ) + (
                    len(in_progress_sample['progress']['errors'])
                    if in_progress_sample is not None else 0
                ),
                'error_types': error_types
            }
            temp_checkpoint = checkpoint_file.with_suffix(
                checkpoint_file.suffix + ".tmp"
            )
            with open(temp_checkpoint, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_checkpoint, checkpoint_file)
        
        # Process each sampled observation set
        for idx in range(start_idx, len(sampled_sets)):
            obs_set = sampled_sets[idx]
            
            if verbose:
                print(f"\nSample {idx + 1}/{n_samples}")
                print(f"  Observation set ID: {obs_set['observation_set_id']}")
                print(f"  Number of observations: {obs_set['n_observations']}")
                print(f"  Number of ground truths: {obs_set['n_compatible_expressions']}")
            
            try:
                # Determine number of queries for this observation set
                if n_queries_per_sample is not None:
                    # Use fixed number
                    n_queries = n_queries_per_sample
                else:
                    # Adaptive: use multiplier of ground truths
                    n_gt = obs_set['n_compatible_expressions']
                    n_queries = max(1, int(n_gt * query_multiplier))
                    if verbose:
                        print(f"  Using {n_queries} queries ({query_multiplier}x {n_gt} ground truths)")

                resume_state = None
                if current_sample is not None:
                    if current_sample.get('n_queries') != n_queries:
                        raise ValueError(
                            "Checkpoint query count does not match this run."
                        )
                    resume_state = current_sample['progress']

                def persist_sample_progress(progress):
                    save_checkpoint({
                        'sample_index': idx,
                        'observation_set_id': obs_set['observation_set_id'],
                        'n_queries': n_queries,
                        'progress': progress
                    })
                
                # Evaluate on this observation set
                result = self.evaluate_single_observation_set(
                    llm,
                    obs_set,
                    n_queries,
                    verbose=False,
                    max_retries=max_retries,
                    resume_state=resume_state,
                    progress_callback=persist_sample_progress
                )
                current_sample = None
                
                all_results.append(result)
                valid_rates.append(result['valid_rate'])
                novelty_rates.append(result['novelty_rate'])
                recovery_rates.append(result['recovery_rate'])
                
                # Aggregate token usage and costs
                if 'token_usage' in result:
                    total_prompt_tokens += result['token_usage']['prompt_tokens']
                    total_completion_tokens += result['token_usage']['completion_tokens']
                    total_tokens += result['token_usage']['total_tokens']
                if 'cost' in result:
                    total_cost += result['cost']
                
                # Aggregate errors
                if 'errors' in result and result['errors']:
                    all_errors.extend(result['errors'])
                    # Update error type counts
                    if 'error_summary' in result:
                        for error_type, count in result['error_summary']['error_types'].items():
                            total_error_counts[error_type] = total_error_counts.get(error_type, 0) + count
                
                if verbose:
                    print(f"  Valid rate: {result['valid_rate']:.2%}")
                    print(f"  Novelty rate: {result['novelty_rate']:.2%}")
                    print(f"  Recovery rate: {result['recovery_rate']:.2%}")
                
                save_checkpoint()

            except Exception as e:
                print(f"  Error processing sample {idx + 1}: {str(e)}")
                traceback.print_exc()
                break
        
        # Calculate statistics
        def calculate_stats(rates):
            if not rates:
                return {'mean': 0, 'std': 0, 'var': 0, 'min': 0, 'max': 0}
            return {
                'mean': np.mean(rates),
                'std': np.std(rates),
                'var': np.var(rates),
                'min': np.min(rates),
                'max': np.max(rates)
            }
        
        # Calculate p-values (one-sample t-test against null hypothesis of 0)
        def calculate_p_value(rates):
            if not rates or len(rates) < 2:
                return None
            t_stat, p_val = stats.ttest_1samp(rates, 0)
            return p_val
        
        # Compile final results
        final_results = {
            'run_id': run_id,
            'llm_name': llm.get_name(),
            'n_samples': len(all_results),
            'n_queries_per_sample': n_queries_per_sample,
            'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
            'query_mode': 'fixed' if n_queries_per_sample is not None else f'adaptive_{query_multiplier}x',
            'seed': seed,
            'max_retries': max_retries,
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'valid_rate': {
                    **calculate_stats(valid_rates),
                    'p_value': calculate_p_value(valid_rates)
                },
                'novelty_rate': {
                    **calculate_stats(novelty_rates),
                    'p_value': calculate_p_value(novelty_rates)
                },
                'recovery_rate': {
                    **calculate_stats(recovery_rates),
                    'p_value': calculate_p_value(recovery_rates)
                }
            },
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens,
                'avg_tokens_per_sample': total_tokens / len(all_results) if all_results else 0,
                'avg_tokens_per_query': total_tokens / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'cost': {
                'total_cost': total_cost,
                'avg_cost_per_sample': total_cost / len(all_results) if all_results else 0,
                'avg_cost_per_query': total_cost / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'error_summary': {
                'total_errors': len(all_errors),
                'error_types': total_error_counts,
                'error_rate': len(all_errors) / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'per_sample_results': all_results
        }
        final_results['method'] = 'improved'
        final_results['method_details'] = (
            'Only valid, in-space, structurally unique candidates are fed back; '
            'candidate diagnostics use visible observations and constraints only.'
        )
        final_results['run_complete'] = len(all_results) == len(sampled_sets)
        
        # Print summary
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 50)
        print(f"Samples evaluated: {len(all_results)}/{n_samples}")
        
        for metric_name, metric_key in [('Valid Rate', 'valid_rate'), 
                                        ('Novelty Rate', 'novelty_rate'), 
                                        ('Recovery Rate', 'recovery_rate')]:
            stats_dict = final_results['statistics'][metric_key]
            print(f"\n{metric_name}:")
            print(f"  Mean ± Std: {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f}")
            print(f"  Variance: {stats_dict['var']:.3f}")
            print(f"  Range: [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
            if stats_dict['p_value'] is not None:
                print(f"  p-value: {stats_dict['p_value']:.4f}")
        
        # Print token usage and cost summary
        print(f"\nToken Usage:")
        print(f"  Total tokens: {final_results['token_usage']['total_tokens']:,}")
        print(f"  Prompt tokens: {final_results['token_usage']['prompt_tokens']:,}")
        print(f"  Completion tokens: {final_results['token_usage']['completion_tokens']:,}")
        print(f"  Avg tokens/sample: {final_results['token_usage']['avg_tokens_per_sample']:.1f}")
        print(f"  Avg tokens/query: {final_results['token_usage']['avg_tokens_per_query']:.1f}")
        
        print(f"\nCost:")
        print(f"  Total cost: ${final_results['cost']['total_cost']:.4f}")
        print(f"  Avg cost/sample: ${final_results['cost']['avg_cost_per_sample']:.4f}")
        print(f"  Avg cost/query: ${final_results['cost']['avg_cost_per_query']:.6f}")
        
        # Print error summary if there were errors
        if final_results['error_summary']['total_errors'] > 0:
            print(f"\nErrors:")
            print(f"  Total errors: {final_results['error_summary']['total_errors']}")
            print(f"  Error rate: {final_results['error_summary']['error_rate']:.2%}")
            if final_results['error_summary']['error_types']:
                print(f"  Error types:")
                for error_type, count in final_results['error_summary']['error_types'].items():
                    print(f"    - {error_type}: {count}")
        
        print("=" * 50)
        return final_results


def setup_llm(llm_type: str, **kwargs) -> LLMInterface:
    """Set up the LLM interface based on type."""
    
    if llm_type == "openai":
        api_key = (
            os.environ.get('OPENAI_API_KEY')
            or os.environ.get('DEEPSEEK_API_KEY')
            or kwargs.get('api_key')
        )
        if not api_key:
            raise ValueError(
                "API key required: set OPENAI_API_KEY or DEEPSEEK_API_KEY"
            )
        
        llm = OpenAILLM(
            model=kwargs.get('model', 'gpt-4'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7),
            base_url=kwargs.get('base_url', 'https://api.deepseek.com')
        )
        llm.client = llm.client.with_options(max_retries=0)
        return llm
    
    elif llm_type == "anthropic":
        api_key = kwargs.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key required")
        
        return AnthropicLLM(
            model=kwargs.get('model', 'claude-3-opus-20240229'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )
    
    elif llm_type == "openrouter":
        api_key = kwargs.get('api_key') or os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OpenRouter API key required")
        
        return OpenRouterLLM(
            model=kwargs.get('model', 'anthropic/claude-3.5-sonnet'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )
    
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description="Run improved Boolean discovery benchmark with sampling")
    parser.add_argument("--dataset", required=True, help="Path to complete Boolean dataset JSON file")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of observation sets to sample")
    parser.add_argument("--n-queries", type=int, default=None, help="Fixed number of queries per observation set (if not set, uses adaptive)")
    parser.add_argument("--query-multiplier", type=float, default=1.0, help="Multiplier for adaptive queries (n_queries = n_gt * multiplier)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Maximum API attempts per query; default 1 limits extra API usage on failures"
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Stable checkpoint ID; reuse the same ID and settings to resume an interrupted run"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path for final results JSON (use a unique path for each seed)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Checkpoint directory; defaults to a separate checkpoints/improved directory"
    )
    
    args = parser.parse_args()
    if args.max_retries < 1:
        parser.error("--max-retries must be at least 1")
    
    # Load configuration
    config = load_config(args.config)
    llm_type = config.get('llm', {}).get('type', 'openrouter')
    
    model = config.get('llm', {}).get('models', {}).get(llm_type)
    if not model:
        default_models = {
            'openrouter': 'openai/gpt-3.5-turbo',
            'openai': 'gpt-4',
            'anthropic': 'claude-3-opus-20240229'
        }
        model = default_models.get(llm_type)
    
    api_key = config.get('llm', {}).get('api_keys', {}).get(llm_type)
    env_vars = {
        'openai': ('OPENAI_API_KEY', 'DEEPSEEK_API_KEY'),
        'anthropic': ('ANTHROPIC_API_KEY',),
        'openrouter': ('OPENROUTER_API_KEY',)
    }
    for env_var in env_vars.get(llm_type, ()):
        if os.environ.get(env_var):
            api_key = os.environ[env_var]
            break
    base_url = config.get('llm', {}).get('base_url', 'https://api.deepseek.com')
    
    temperature = config.get('llm', {}).get('temperature', 0.7)
    configured_checkpoint_dir = config.get('benchmark', {}).get('checkpoint', 'checkpoints')
    checkpoint_dir = args.checkpoint_dir or str(
        Path(configured_checkpoint_dir) / "improved"
    )
    run_id = args.run_id or config.get('benchmark', {}).get('run_id', None)
    if run_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]+", str(run_id)):
        parser.error("--run-id may contain only letters, numbers, underscores, and hyphens")
    verbose = config.get('benchmark', {}).get('verbose', True)

    if not Path(args.dataset).exists():
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)
    dataset_name = Path(args.dataset).stem
    model_name = Path(model).stem if model else llm_type
    output_pattern = config.get('benchmark', {}).get('output_pattern', 'results_{dataset_name}_{model}.json')
    baseline_output = Path(output_pattern.format(
        dataset_name=dataset_name, model=model_name, llm_type=llm_type
    ))
    improved_default_output = baseline_output.with_name(
        f"{baseline_output.stem}_improved{baseline_output.suffix}"
    )
    output = args.output or str(improved_default_output)

    # Generate output filename if not specified
    if output is None:
        dataset_name = Path(args.dataset).stem
        model_name = Path(model).stem if model else llm_type
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"results/{dataset_name}_{model_name}_{timestamp}.json"

    # Initialize benchmark
    benchmark = BooleanBenchmarkImproved(args.dataset)
     
    # Print configuration
    print("\n" + "=" * 60)
    print("IMPROVED BOOLEAN BENCHMARK CONFIGURATION")
    print("Method: improved (validated unique archive + verification feedback)")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"LLM Type: {llm_type}")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}")
    print(f"Samples: {args.n_samples}")
    print(f"Queries per sample: {args.n_queries}")
    print(f"Maximum API attempts per query: {args.max_retries}")
    print(f"Seed: {args.seed}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Output: {output}")
    print("=" * 60)
    
    # Set up LLM
    llm = setup_llm(
        llm_type,
        model=model,
        api_key=api_key,
        temperature=temperature,
        base_url=base_url
    )
    
    # Run benchmark
    results = benchmark.run_benchmark(
        llm=llm,
        n_samples=args.n_samples,
        n_queries_per_sample=args.n_queries,
        query_multiplier=args.query_multiplier,
        seed=args.seed,
        verbose=verbose,
        checkpoint_dir=checkpoint_dir,
        run_id=run_id,
        max_retries=args.max_retries
    )
    
    if not results['run_complete']:
        print(
            "\nRun incomplete. The final results file was not written. "
            "Rerun with the same seed, run-id, query settings, and checkpoint "
            "directory to resume from saved per-response progress."
        )
        sys.exit(2)

    # Save final results only after every sampled observation set is complete.
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFinal results saved to: {output}")


if __name__ == "__main__":
    main()