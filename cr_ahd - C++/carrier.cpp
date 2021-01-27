#include <vector>
#include <memory>

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

carrier::carrier(nlohmann::ordered_json carrier_json)
{
	id_ = carrier_json.at("id_");
	depot_ = vertex(carrier_json.at("depot"));
	//std::vector<vehicle> vehicles_;
	
	for (auto vehicle_json : carrier_json.at("vehicles")) {
		auto v = std::make_unique<vehicle>(vehicle_json);
		vehicles_.push_back(std::move(v));
	}
	
	// TODO assigned requests: (a) pointer to instance's request (b) copy, redundant to instances requests (c) instance's requests point to carriers' requests
	// implementing (b); although naive, requests only contain built-in types and are a fairly simple class

	for (auto request_json : carrier_json.at("requests")) {
		auto r = std::make_unique<vertex>(request_json);
		requests_.push_back(std::move(r));
		unrouted_.push_back(std::move(r));
	}
}

std::ostream& operator<<(std::ostream& os, carrier& carrier)
{
	os << "Carrier " << carrier.id_ << " {" << carrier.depot_.get_id() << "; " << carrier.num_vehicles() << " vehicles; " << carrier.num_requests() << " requests; " << carrier.num_unrouted() << " unrouted}";
	return os;
}

carrier::~carrier()
{
	std::cout << id_ << " destroyed!\n";
}



int carrier::num_vehicles(){return vehicles_.size();}

int carrier::num_requests(){return requests_.size();}

int carrier::num_unrouted(){return unrouted_.size();}

const std::vector<std::unique_ptr<vertex>>& carrier::get_requests() {	return requests_;}


