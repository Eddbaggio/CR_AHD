#include "carrier.h"
#include  "vertex.h"
#include "utils.h"

carrier::carrier()
{
}

carrier::carrier(
	const std::string& id,
	const vertex depot,
	//std::array<Vehicle, ? ? > vehicles
	const std::map<std::string, vertex> requests
)
	: id_(id), depot_(depot), /*vehicles_(vehicles),*/ requests_(requests)
{}

void carrier::assign_request(vertex request)
{
	// extend the map of requests and also the map of unrouted requests by the given request
}


