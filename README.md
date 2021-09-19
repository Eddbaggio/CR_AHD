# CR_AHD
Advanced Methods for Optimization - Pickup and Delivery Problem with Dynamic Time Window Assignment

To run the code, the following external packages must be installed to a **Python 3.8** environment:
* pandas 
* numpy
* ploty
* tqdm
* scipy


Then, make sure the path in [main.py at line 22](https://github.com/selting/CR_AHD/blob/be45827a50afc75db7c0e8ffa151389a4599b900/cr_ahd%20-%20Python/src/cr_ahd/main.py#L22) is correct and points to the directory of input data.
From [line 25](https://github.com/selting/CR_AHD/blob/be45827a50afc75db7c0e8ffa151389a4599b900/cr_ahd%20-%20Python/src/cr_ahd/main.py#L25) to 28, a specific or a random file can be selected if desired. Currently, all files are solved.
Running main.py will solve all specified files using the parameters set in [`param_gen.py`](https://github.com/selting/CR_AHD/blob/be45827a50afc75db7c0e8ffa151389a4599b900/cr_ahd%20-%20Python/src/cr_ahd/solver_module/param_gen.py) by creating a [Solver](https://github.com/selting/CR_AHD/blob/be45827a50afc75db7c0e8ffa151389a4599b900/cr_ahd%20-%20Python/src/cr_ahd/solver_module/solver.py) that is defining the flow of solving steps. 

Much work is done by these three classes: Instance, CAHDSolution and Tour, which are all located in the [core module](https://github.com/selting/CR_AHD/tree/Advanced_Methods_in_Optimization/cr_ahd%20-%20Python/src/cr_ahd/core_module)

Moreover, the routing logic, including [tour construction](https://github.com/selting/CR_AHD/blob/be45827a50afc75db7c0e8ffa151389a4599b900/cr_ahd%20-%20Python/src/cr_ahd/routing_module/tour_construction.py), [metaheuristics](https://github.com/selting/CR_AHD/blob/be45827a50afc75db7c0e8ffa151389a4599b900/cr_ahd%20-%20Python/src/cr_ahd/routing_module/metaheuristics.py) and [neighborhoods](https://github.com/selting/CR_AHD/blob/be45827a50afc75db7c0e8ffa151389a4599b900/cr_ahd%20-%20Python/src/cr_ahd/routing_module/neighborhoods.py) is located in the [routing module](https://github.com/selting/CR_AHD/tree/Advanced_Methods_in_Optimization/cr_ahd%20-%20Python/src/cr_ahd/routing_module)
