import viktor as vkt

from src import genetic_optimize


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
