import copy
from io import StringIO
from typing import List, Dict, Any, Optional, Tuple

import pygad
import numpy as np
import viktor as vkt

# =============================================================================
# Constants for Field Types
# =============================================================================

NUMBER_TYPE = "number"
OPTION_TYPE = "option"
BOOLEAN_TYPE = "boolean"
ARRAY_LENGTH_TYPE = "array_length"
ARRAY_NUMBER_TYPE = "array_number"
ARRAY_OPTION_TYPE = "array_option"
ARRAY_BOOLEAN_TYPE = "array_boolean"
TABLE_NUMBER_TYPE = "table_number"
TABLE_OPTION_TYPE = "table_option"
TABLE_BOOLEAN_TYPE = "table_boolean"

# =============================================================================
# Helper Functions
# =============================================================================

def _serialize_options(options: list) -> list:
    """Convert Viktor OptionListElements to their values for serialization."""
    serializable_options = []
    for option in options:
        if isinstance(option, vkt.OptionListElement):
            serializable_options.append(option.value)
        else:
            serializable_options.append(option)
    return serializable_options

def _handle_field(
    field_obj: Any,
    field_name: str,
    params: dict,
    context: str = "root",
    index: Optional[int] = None,
    parent_name: Optional[str] = None
) -> Tuple[Any, dict]:
    """
    Convert a Viktor field to a gene space entry and mapping.
    Handles NumberField, OptionField, BooleanField.
    """
    if isinstance(field_obj, vkt.NumberField):
        gene = {"low": field_obj._min, "high": field_obj._max, "step": field_obj._step}
        if context == "root":
            mapping = {"name": field_name, "type": NUMBER_TYPE}
        elif context == "array":
            mapping = {
                "name": f"{parent_name}[{index}].{field_name}",
                "type": ARRAY_NUMBER_TYPE,
                "array_name": parent_name,
                "array_index": index,
                "field_name": field_name,
            }
        else:  # context == "table"
            mapping = {
                "name": f"{parent_name}[{index}].{field_name}",
                "type": TABLE_NUMBER_TYPE,
                "table_name": parent_name,
                "table_index": index,
                "field_name": field_name,
            }
        return gene, mapping

    elif isinstance(field_obj, vkt.OptionField):
        options = field_obj._options if not field_obj._dynamic_options else field_obj._dynamic_options(params)
        serializable_options = _serialize_options(options)
        gene = list(range(len(options)))
        if context == "root":
            mapping = {"name": field_name, "type": OPTION_TYPE, "options": serializable_options}
        elif context == "array":
            mapping = {
                "name": f"{parent_name}[{index}].{field_name}",
                "type": ARRAY_OPTION_TYPE,
                "array_name": parent_name,
                "array_index": index,
                "field_name": field_name,
                "options": serializable_options,
            }
        else:  # context == "table"
            mapping = {
                "name": f"{parent_name}[{index}].{field_name}",
                "type": TABLE_OPTION_TYPE,
                "table_name": parent_name,
                "table_index": index,
                "field_name": field_name,
                "options": serializable_options,
            }
        return gene, mapping

    elif isinstance(field_obj, vkt.BooleanField):
        gene = [0, 1]
        if context == "root":
            mapping = {"name": field_name, "type": BOOLEAN_TYPE}
        elif context == "array":
            mapping = {
                "name": f"{parent_name}[{index}].{field_name}",
                "type": ARRAY_BOOLEAN_TYPE,
                "array_name": parent_name,
                "array_index": index,
                "field_name": field_name,
            }
        else:  # context == "table"
            mapping = {
                "name": f"{parent_name}[{index}].{field_name}",
                "type": TABLE_BOOLEAN_TYPE,
                "table_name": parent_name,
                "table_index": index,
                "field_name": field_name,
            }
        return gene, mapping

    # Not a handled field type
    return None, None

# =============================================================================
# Main Conversion Functions
# =============================================================================

def create_gene_space_from_parametrization(
    parametrization_class: Any,
    path: str = "",
    params: Optional[dict] = None
) -> Tuple[List[Any], Dict[int, dict]]:
    """
    Convert a Viktor parametrization class to a pygad gene space and mapping.

    Args:
        parametrization_class: Viktor parametrization class instance
        path: Dot-separated path to a section (e.g., "calculation_params")
        params: Parameter values for dynamic options

    Returns:
        Tuple of (gene_space, field_mapping)
    """
    if params is None:
        params = {}

    # Traverse to the correct section if a path is given
    if path == "":
        input_fields = parametrization_class.__fields__
        current_params = parametrization_class
    else:
        current_params = parametrization_class
        try:
            for sub_path in path.split("."):
                current_params = getattr(current_params, sub_path)
            input_fields = [n for n in list(current_params._attrs.keys()) if not n.startswith("_")]
        except AttributeError:
            raise vkt.UserError("Wrong input; path should be a Section, Page or Step")

    gene_space = []
    field_mapping = {}

    for field_name in input_fields:
        field_obj = getattr(current_params, field_name)
        gene, mapping = _handle_field(field_obj, field_name, params)
        if gene is not None:
            gene_space.append(gene)
            field_mapping[len(gene_space) - 1] = mapping
        elif isinstance(field_obj, vkt.DynamicArray):
            # Handle dynamic arrays (lists of fields)
            array_gene_space, array_mapping = _handle_dynamic_array(field_obj, field_name, params)
            offset = len(gene_space)
            gene_space.extend(array_gene_space)
            for k, v in array_mapping.items():
                field_mapping[offset + k] = v
        elif isinstance(field_obj, vkt.Table):
            # Handle tables (lists of fields with fixed length)
            table_gene_space, table_mapping = _handle_table(field_obj, field_name, path, params)
            offset = len(gene_space)
            gene_space.extend(table_gene_space)
            for k, v in table_mapping.items():
                field_mapping[offset + k] = v

    return gene_space, field_mapping

def _handle_dynamic_array(
    array_field: Any,
    array_name: str,
    params: dict
) -> Tuple[List[Any], Dict[int, dict]]:
    """
    Handle DynamicArray fields by creating a flattened gene space.

    Returns:
        Tuple of (gene_space, field_mapping)
    """
    gene_space = []
    field_mapping = {}

    # First gene: array length
    if None in [array_field._min, array_field._max]:
        raise vkt.UserError("For optimizations, set min and max arguments of the DynamicArray")
    gene_space.append(list(range(array_field._min, array_field._max + 1)))
    field_mapping[0] = {"name": f"{array_name}_length", "type": ARRAY_LENGTH_TYPE, "array_name": array_name}

    # Genes for each possible array element
    array_fields = [n for n in list(array_field._attrs.keys()) if not n.startswith("_")]
    idx = 1
    for i in range(array_field._max):
        for field_name in array_fields:
            field_obj = getattr(array_field, field_name)
            gene, mapping = _handle_field(field_obj, field_name, params, context="array", index=i, parent_name=array_name)
            if gene is not None:
                gene_space.append(gene)
                field_mapping[idx] = mapping
                idx += 1
    return gene_space, field_mapping

def _handle_table(
    table_field: Any,
    table_name: str,
    path: str,
    params: dict
) -> Tuple[List[Any], Dict[int, dict]]:
    """
    Handle Table fields by creating a flattened gene space.

    Returns:
        Tuple of (gene_space, field_mapping)
    """
    gene_space = []
    field_mapping = {}

    # Traverse params to the correct table section
    table_params = params
    for folder in path.split(".") + [table_name]:
        table_params = table_params.get(folder, {})
    array_fields = [n for n in list(table_field._attrs.keys()) if not n.startswith("_")]
    idx = 0
    for i in range(len(table_params)):
        for field_name in array_fields:
            field_obj = getattr(table_field, field_name)
            gene, mapping = _handle_field(field_obj, field_name, params, context="table", index=i, parent_name=table_name)
            if gene is not None:
                gene_space.append(gene)
                field_mapping[idx] = mapping
                idx += 1
    return gene_space, field_mapping

# =============================================================================
# Utility Functions
# =============================================================================

def build_nested_dict(path: str, value: Any) -> dict:
    """
    Recursively build a nested dictionary from a dotted path and a value.
    """
    keys = path.split(".")
    if path == "":
        return value
    if len(keys) == 1:
        try:
            return {keys[0]: round(float(value), 2)}
        except (TypeError, ValueError):
            return {keys[0]: value}
    return {keys[0]: build_nested_dict(".".join(keys[1:]), value)}

def set_nested_attr(obj: Any, attr_path: str, value: Any) -> None:
    """
    Set a nested attribute on an object using a dotted path.
    """
    attrs = attr_path.split(".")
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)

def decode_solution_to_parameters(
    solution: List[Any],
    field_mapping: Dict[int, dict],
    path: str
) -> dict:
    """
    Convert a GA solution (list of gene values) back to Viktor parameters format.
    """
    params = {}
    for gene_idx, gene_value in enumerate(solution):
        if gene_idx not in field_mapping:
            continue
        field_info = field_mapping[gene_idx]
        field_type = field_info["type"]

        if field_type == NUMBER_TYPE:
            params[field_info["name"]] = float(gene_value)
        elif field_type == OPTION_TYPE:
            val = field_info["options"][int(gene_value)]
            if isinstance(val, vkt.OptionListElement):
                params[field_info["name"]] = val.value
            else:
                params[field_info["name"]] = val
        elif field_type == BOOLEAN_TYPE:
            params[field_info["name"]] = bool(gene_value)
        elif field_type == ARRAY_LENGTH_TYPE:
            array_name = field_info["array_name"]
            if array_name not in params:
                params[array_name] = []
            params[f"__{array_name}_length"] = gene_value
        elif field_type.startswith("array_"):
            array_name = field_info["array_name"]
            array_index = field_info["array_index"]
            field_name = field_info["field_name"]
            actual_length = params.get(f"__{array_name}_length", 0)
            if array_index >= actual_length:
                continue
            if array_name not in params:
                params[array_name] = []
            while len(params[array_name]) <= array_index:
                params[array_name].append({})
            if field_type == ARRAY_NUMBER_TYPE:
                params[array_name][array_index][field_name] = float(gene_value)
            elif field_type == ARRAY_OPTION_TYPE:
                params[array_name][array_index][field_name] = field_info["options"][int(gene_value)]
            elif field_type == ARRAY_BOOLEAN_TYPE:
                params[array_name][array_index][field_name] = bool(gene_value)
        elif field_type.startswith("table_"):
            table_name = field_info["table_name"]
            table_index = field_info["table_index"]
            field_name = field_info["field_name"]
            if table_name not in params:
                params[table_name] = []
            while len(params[table_name]) <= table_index:
                params[table_name].append({})
            if field_type == TABLE_NUMBER_TYPE:
                params[table_name][table_index][field_name] = float(gene_value)
            elif field_type == TABLE_OPTION_TYPE:
                params[table_name][table_index][field_name] = field_info["options"][int(gene_value)]
            elif field_type == TABLE_BOOLEAN_TYPE:
                params[table_name][table_index][field_name] = bool(gene_value)
    # Remove temporary length fields
    params = {k: v for k, v in params.items() if not k.startswith("__")}
    return params