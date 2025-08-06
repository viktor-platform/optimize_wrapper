# Optimize Wrapper

A VIKTOR wrapper to wrap your optimization function and make it the fitness function for a Genetic Algorithm.

## Installation

You can install this package directly from GitHub:

```bash
pip install git+https://github.com/viktor-platform/optimize_wrapper.git
```

## Usage

```python
import viktor as vkt

from optimize_wrapper import genetic_optimize


class Parametrization(vkt.Parametrization):
    x = vkt.NumberField("X")
    y = vkt.NumberField("Y")

class Controller(vkt.Controller):
    parametrization = Parametrization

    @genetic_optimize(
        optimized_parameters=Parametrization,
        path="areas_section",
        sol_per_pop=250,
        num_generations=100,
        amount_of_solutions=100
    )
    def optimize_x_and_y(self, optimized_params, params, **kwargs):
        return optimized_params.x + optimized_params.y

```

## Requirements

- Python >= 3.8
- munch >= 4.0.0
- pygad >= 3.5.0
- viktor >= 14.24.0
- matplotlib
- numpy

## Development

To install in development mode:

```bash
git clone https://github.com/viktor-platform/optimize_wrapper.git
cd optimize_wrapper
pip install -e .
```

## License

This project is part of the VIKTOR platform.
