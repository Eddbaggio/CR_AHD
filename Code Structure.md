# Classes
## Instance
1. Attributes
	* ID
	* Requests
	* Carriers ? 
	<!-- For carriers to be part of the instance, the request-to-carrier assignment must already be encoded in the instance file rather than happening on the fly -->
	* Distance Matrix (all depots and all customers)
2. Methods
	* init (read from csv) -> creates carriers (incl. Vehicles) & requests & distance matrix
	* distance(i,j) -> get distance between two requests i, j -> int/float
	* time(i,j) -> get travel time between two requests i,j -> int/float/
	* 

## Request
1. Attributes
	* ID
	* coordinates (x,y)
	* time window (e,l)
	* [load q]
2. Methods
	* assign to carrier

## Carrier
1. Attributes
	* ID
	* Depot location
	* Vehicles?
	* Requests?
	* incumbent solution
2. Methods

## Vehicle
1. Attributes
	* ID
	* [Load Capacity]
	* [Time Capacity]
2. Methods

## Candidate Solution
1. Attributes
	* ID
	* Adjacency Representation of VRPTW solution ???
	* Objective Function Evaluation / Objective Value (based on the solution representation this may require a decoder)
2. Methods
	* [LS Move -> e.g. swap, insert, ...]
	* Evaluate -> compute obj. function value
	* Check feasibility -> ensure that all constraints are satisfied
	* print -> readable console output of solution
	* plot -> plotting the solution


# Misc
1. Global Options
	* log: bool (write console output to file?)
	* verbose: int (specify level of detail for console output)
	* plot_options: 
		* plot: bool (create plots of final/intermediate solution?)
		* plot size
		* plot style
		* ...
