import matplotlib.pyplot as plt
from io import StringIO


def create_detailed_image_result(data, chosen, gene_space, field_mapping, show_values=True):
    """
    Create a detailed SVG plot of genetic algorithm optimization results.
    Shows all evaluated solutions in gray and chosen (optimal) solutions in green.
    Annotates values for chosen solutions if there are few, and provides meaningful axis labels.
    Args:
        data (list of list): All evaluated solutions.
        chosen (list of list): Chosen/optimal solutions.
        gene_space (list): Parameter space for each gene.
        field_mapping (dict): Mapping from gene index to field info dict.
        show_values (bool): Whether to annotate chosen solution values.
    Returns:
        StringIO: SVG image data.
    """
    fig, ax = plt.subplots(figsize=(16, 10))

    def get_gene_label(gene_idx):
        """Return a meaningful label for a gene based on field mapping."""
        if gene_idx in field_mapping:
            field_info = field_mapping[gene_idx]
            field_type = field_info["type"]
            name = field_info["name"]

            # Clean up the name for display
            if field_type.startswith("array_"):
                # For array fields, show a shorter version
                if "[" in name and "]" in name:
                    # Extract array name and field
                    parts = name.split("[")
                    array_name = parts[0]
                    rest = "[" + parts[1]
                    if "." in rest:
                        field_part = rest.split(".")[-1]
                        return f"{array_name}[].{field_part}"
                return name
            if field_type == "array_length":
                return f"{field_info['array_name']}_len"
            else:
                return name
        return f"Gene_{gene_idx + 1}"

    def normalize_and_get_display_value(value, gene_idx):
        """Normalize value for plotting and return display string."""
        if gene_idx >= len(gene_space):
            return 50, str(value)

        space = gene_space[gene_idx]
        field_info = field_mapping.get(gene_idx, {})
        field_type = field_info.get("type", "unknown")

        if isinstance(space, dict) and "low" in space and "high" in space:
            # Numeric field
            low, high = space["low"], space["high"]
            if high == low:
                return 50, f"{value:.1f}"
            normalized = ((value - low) / (high - low)) * 100
            return normalized, f"{value:.1f}"

        elif isinstance(space, list):
            # Categorical field
            if len(space) <= 1:
                return 50, str(value)
            max_idx = len(space) - 1
            try:
                idx = int(value)
                normalized = (idx / max_idx) * 100
                # Get display value
                if field_type in ["option", "array_option"]:
                    options = field_info.get("options", [])
                    if 0 <= idx < len(options):
                        display_val = options[idx]
                    else:
                        display_val = f"idx_{idx}"
                elif field_type in ["boolean", "array_boolean"]:
                    display_val = "True" if idx else "False"
                else:
                    display_val = str(idx)
                return normalized, display_val
            except (ValueError, TypeError):
                return 0, str(value)
        return 50, str(value)

    # Prepare normalized data and display labels
    def process_rows(rows):
        percent_rows, label_rows = [], []
        for row in rows:
            percent_row, label_row = [], []
            for i, val in enumerate(row):
                norm_val, display_val = normalize_and_get_display_value(val, i)
                percent_row.append(norm_val)
                label_row.append(display_val)
            percent_rows.append(percent_row)
            label_rows.append(label_row)
        return percent_rows, label_rows

    data_percent, _data_labels = process_rows(data)
    chosen_percent, chosen_labels = process_rows(chosen)

    # Plot all evaluated solutions in light gray
    for row in data_percent:
        ax.plot(range(len(row)), row, color="lightgray", alpha=0.4, linewidth=0.8)

    # Plot chosen solutions in green
    for i, row in enumerate(chosen_percent):
        ax.plot(
            range(len(row)), row, color="green", alpha=0.9, linewidth=3,
            label=f"Solution {i + 1}" if i < 5 else ""
        )  # Label first 5 solutions

    # Annotate values for chosen solutions (if not too many)
    if show_values and len(chosen_percent) <= 3:
        for i, (row, labels) in enumerate(zip(chosen_percent, chosen_labels)):
            for j, (y_val, label) in enumerate(zip(row, labels)):
                ax.annotate(
                    label, (j, y_val), xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7, color='darkgreen'
                )

    # Axis properties
    num_genes = len(gene_space)
    ax.set_xlim(-0.5, num_genes - 0.5)
    ax.set_xticks(range(num_genes))
    ax.set_xticklabels(
        [get_gene_label(i) for i in range(num_genes)],
        rotation=45, 
        ha="right", 
        fontsize=10
    )
    ax.set_ylim(-5, 105)
    ax.set_ylabel("Normalized Value (0-100%)", fontsize=12)
    ax.set_xlabel("Parameters", fontsize=12)

    # Reference lines and grid
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.4)
    ax.axhline(y=50, color='blue', linestyle='--', alpha=0.3)
    ax.axhline(y=100, color='black', linestyle='-', alpha=0.4)
    ax.grid(True, alpha=0.2, axis='y')

    # Title and legend
    title = (
        f"Genetic Algorithm Optimization Results\n"
        f"{len(data)} solutions evaluated, {len(chosen)} optimal solutions shown"
    )
    ax.set_title(title, fontsize=14, pad=20)
    if len(chosen_percent) <= 10:
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    plt.tight_layout()

    # Save to SVG in memory
    svg_data = StringIO()
    fig.savefig(svg_data, format="svg", bbox_inches='tight', dpi=300)
    plt.close(fig)
    return svg_data
