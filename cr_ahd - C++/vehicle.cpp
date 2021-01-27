#include "vehicle.h"

vehicle::vehicle(std::string id, int capacity)
	:id_{ id }, capacity_{ capacity }
{
}

vehicle::vehicle(nlohmann::ordered_json vehicle_json)
{
	id_ = vehicle_json.at("id_");
	//capacity_ = vehicle_json.at("capacity");	//implemented with NULL in Python
}



std::ostream& operator<<(std::ostream os, const vehicle& vehicle)
{
	os << "Vehicle " << vehicle.id_ <<": " <<"Capacity = " << vehicle.capacity_;
	return os;
}
