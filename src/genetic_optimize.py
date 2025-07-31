import copy
from io import StringIO
from typing import List, Dict, Any, Mapping

from src.optimization_image import create_detailed_image_result
import pygad
import numpy as np
import viktor as vkt


# HELPER FUNCTIONS
from munch import munchify

from src.process_parametrization import create_gene_space_from_parametrization, decode_solution_to_parameters


def set_value_by_path(
    existing_dict: Mapping[str, Any],
    path: str,
    value: Any,
    *,
    separator: str = ".",
    create_missing: bool = False,
    mutate_original: bool = False,
) -> Dict[str, Any]:
    """
    Set (or replace) the value at a dotted‐path location inside a nested dictionary.

    Parameters
    ----------
    existing_dict : Mapping[str, Any]
        The starting dictionary.
    path : str
        Dotted path to the key you want to set, e.g. ``"path.to.values"``.
    value : Any
        The new value to assign.
    separator : str, optional
        The character that separates path components.  Defaults to ``"."``.
    create_missing : bool, optional
        If ``True`` (default ``False``), create intermediate dictionaries when they
        don’t exist instead of raising a ``KeyError``.
    mutate_original : bool, optional
        If ``True``, update ``existing_dict`` in place and return it; otherwise a
        deep copy is made first and the copy is returned.

    Returns
    -------
    Dict[str, Any]
        The dictionary with the value set at the requested path.

    Raises
    ------
    KeyError
        If an intermediate key is missing and ``create_missing`` is ``False``.
    """
    if mutate_original:
        target = existing_dict  # keep original reference
    else:
        target = copy.deepcopy(existing_dict)

    parts = path.split(separator)
    if not parts:
        raise ValueError("Path must contain at least one key")

    cursor = target
    for key in parts[:-1]:
        if key in cursor:
            if isinstance(cursor[key], dict):
                cursor = cursor[key]
            else:
                raise TypeError(
                    f"Encountered non‑dict node at '{key}' while traversing '{path}'"
                )
        else:
            if create_missing:
                cursor[key] = {}
                cursor = cursor[key]
            else:
                raise KeyError(
                    f"Key '{key}' not found while traversing '{path}'. "
                    "Use create_missing=True to create intermediate nodes."
                )

    # Set (or overwrite) the final key
    cursor[parts[-1]] = value
    return target


def are_rows_similar(
    row1: np.ndarray, row2: np.ndarray, thresholds: List[float]
) -> bool:
    """Check if two rows are similar based on column-wise Euclidean distances."""
    for col, threshold in enumerate(thresholds):
        # Euclidean dist
        if np.linalg.norm(row1[col] - row2[col]) > threshold:
            return False
    return True


def filter_similar_rows(array: np.ndarray, thresholds: List[float]) -> np.ndarray:
    """Filter out similar rows based on column-wise Euclidean distances."""
    filtered_indices = []

    for i, row in enumerate(array):
        if not any(
            are_rows_similar(row, array[j], thresholds) for j in filtered_indices
        ):
            filtered_indices.append(i)

    return array[filtered_indices]


def calculate_thresholds(gene_space, percentage_threshold=0.02):
    """Calculate thresholds for each gene based on its type"""
    thresholds = []

    for gene in gene_space:
        if isinstance(gene, dict) and "high" in gene and "low" in gene:
            # NumberField
            threshold = (gene["high"] - gene["low"]) * percentage_threshold
        elif isinstance(gene, list):
            # OptionField or BooleanField
            threshold = 0  # Exact match required for discrete values
        else:
            threshold = 0

        thresholds.append(threshold)

    return thresholds


def genetic_optimize(
    optimized_parameters,  # TODO: Parametrization or Group
    num_generations=100,
    num_parents_mating=4,
    sol_per_pop=50,
    amount_of_solutions=10,
    path="",
):
    """Wrapper for an Optimization Function, which runs a genetic algorithm"""

    # Create an instance of the GA class
    def decorator(func):
        cc = 0
        def wrapper(self, params, *args, **kwargs):
            """Create, setup and run the GA instance, with the wrapped function as fitness"""
            # Get all the optimizable parameters
            vkt.progress_message("Starting up Genetic Algorithm", percentage=10)
            gene_space, field_mapping = create_gene_space_from_parametrization(optimized_parameters, path, params)
            good_solutions = []

            def fitness_wrapper(_pygad_instance, solution, _solution_idx):
                """Wrapping the actual function in the Controller class"""
                optimization_params = decode_solution_to_parameters(solution, field_mapping, path)
                solution_params = munchify(set_value_by_path(params, path, optimization_params))
                return func(self, solution_params, params, *args, **kwargs)

            def on_generation(_ga_instance):
                """Subscribe to population generation, to save the results for postprocessing"""
                nonlocal cc
                if cc % 10 == 0:
                    vkt.progress_message("Running optimization", percentage=cc / num_generations * 100)
                population = _ga_instance.population
                fitness = _ga_instance.last_generation_fitness
                sorted_indices = np.argsort(fitness)[::-1]
                good_solutions.extend(population[sorted_indices[:5]])
                cc += 1

            ga_instance = pygad.GA(
                num_generations=num_generations,
                num_parents_mating=num_parents_mating,
                fitness_func=fitness_wrapper,
                num_genes=len(gene_space),
                gene_space=gene_space,
                gene_type=float,
                sol_per_pop=sol_per_pop,
                on_generation=on_generation,
                mutation_probability=0.3,
            )
            ga_instance.run()
            vkt.progress_message("Creating outputs...")

            # Remove duplicates and filter similar results
            unique_solutions = np.unique(np.array(good_solutions), axis=0)

            thresholds = calculate_thresholds(gene_space)
            unique_solutions = filter_similar_rows(
                array=unique_solutions, thresholds=thresholds
            )

            # Calculate fitness for each unique solution
            fitness_values = np.array(
                [
                    fitness_wrapper(ga_instance, solution, idx)
                    for idx, solution in enumerate(unique_solutions)
                ]
            )

            # Sort solutions by fitness (assuming higher fitness is better)
            sorted_indices = np.argsort(fitness_values)[::-1]
            top_solutions = unique_solutions[sorted_indices]
            top_fitness_values = fitness_values[sorted_indices]

            # Print top solutions, where amount is function input (Defaults to 10)
            results = []
            for i in range(min(amount_of_solutions, len(top_solutions))):
                assert len(gene_space) == len(top_solutions[i])
                results_per_gene = decode_solution_to_parameters(top_solutions[i], field_mapping, path)
                results_per_gene = build_nested_dict(path, results_per_gene)
                results_including_fitness = results_per_gene.copy()
                results_including_fitness["fitness"] = round(float(top_fitness_values[i]), 2)
                results.append(
                    vkt.OptimizationResultElement(
                        results_per_gene, results_including_fitness
                    )
                )

            # Create an image, setup headers adn return a OptimizationResult
            # output_headers = {name : name for name in list(gene_space_dict.keys())}
            output_headers = {}
            output_headers["fitness"] = "Fitness"

            image = vkt.ImageResult(create_detailed_image_result(
                data=unique_solutions,
                chosen=top_solutions[:amount_of_solutions],
                gene_space=gene_space,
                field_mapping=field_mapping,
                show_values=True  # Show actual values on the plot
            ))
            return vkt.OptimizationResult(
                results, output_headers=output_headers, image=image
            )

        return wrapper

    return decorator
