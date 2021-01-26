#include <vector>

#include "carrier.h"
#include "vertex.h"
#include "utils.h"
#include "vehicle.h"

carrier::carrier()
{
}

//carrier::carrier(
//	const std::string& id,
//	const vertex depot,
//	std::vector<vehicle> vehicles,
//	const std::map<std::string, vertex> requests
//)
//	: id_(id), depot_(depot), /*vehicles_(vehicles),*/ requests_(requests)
//{}

carrier::carrier(nlohmann::json carrier_json)
{
	id_ = carrier_json.at("id_");
	depot_ = vertex(carrier_json.at("depot"));
	std::vector<vehicle> vehicles_;
	
	for (auto vehicle_json : carrier_json.at("vehicles"))
		vehicles_.push_back(vehicle(vehicle_json));
	
	// TODO assigned requests: (a) pointer to instance's request (b) copy, redundant to instances requests (c) instance's requests point to carriers' requests
	// implementing (b); although naive, requests only contain built-in types and are a fairly simple class
	std::unordered_map<std::string, vertex> requests_;	//initialize empty
	std::unordered_map<std::string, vertex> unrouted_;	//initialize empty
	for (auto request_json : carrier_json.at("requests")){
		requests_.insert(std::pair<std::string, vertex>(request_json.at("id_"), vertex(request_json)));
		unrouted_.insert(std::pair<std::string, vertex>(request_json.at("id_"), vertex(request_json)));
	}

}



void carrier::assign_request(vertex request)
{
	// extend the map of requests and also the map of unrouted requests by the given request
}


