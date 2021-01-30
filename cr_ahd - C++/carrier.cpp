#include <vector>
#include <memory>

#include "carrier.h"
#include "vertex.h"
#include "utils.h"
#include "vehicle.h"


// constructors & destructors
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

carrier::carrier(nlohmann::ordered_json carrier_json)
{
	id_ = carrier_json.at("id_");
	depot_ = vertex(carrier_json.at("depot"));
	//std::vector<vehicle> vehicles_;

	for (auto vehicle_json : carrier_json.at("vehicles")) {
		vehicle v(vehicle_json);
		tour t(vehicle_json.at("id_"), std::vector<vertex>{ depot_, depot_ });	//TODO should i rather pass depot by reference?
		v.set_tour(t);		
		vehicles_.push_back(v);
	}

	// TODO assigned requests: (a) pointer to instance's request (b) copy, redundant to instances requests (c) instance's requests point to carriers' requests
	// implementing (b); although naive, requests only contain built-in types and are a fairly simple class

	for (auto request_json : carrier_json.at("requests")) {
		auto r = vertex(request_json);
		requests_.push_back(r);
		unrouted_.push_back(r);
	}
}

carrier::~carrier() { std::cout << id_ << " destroyed!\n"; }

//setters & getters
const std::vector<vertex>& carrier::get_requests() const { return requests_; }
const std::vector<vehicle>& carrier::get_vehicles() const { return vehicles_; }


//operators
std::ostream& operator<<(std::ostream& os, carrier& carrier)
{
	os << "Carrier " << carrier.id_ << " {" << carrier.depot_.get_id() << "; " << carrier.num_vehicles() << " vehicles; " << carrier.num_requests() << " requests; " << carrier.num_unrouted() << " unrouted}";
	return os;
}

// other member functions

int carrier::num_vehicles() const { return vehicles_.size(); }

int carrier::num_requests() const { return requests_.size(); }

int carrier::num_unrouted() const { return unrouted_.size(); }

//algorithms

void carrier::compute_all_vehicle_cost_and_schedules(const std::map<std::string, std::map<std::string, float>>& distance_matrix)
{
	for (auto& v : vehicles_) {
		v.get_tour().compute_cost_and_schedule(distance_matrix);
	}
}

