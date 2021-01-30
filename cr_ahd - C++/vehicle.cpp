#include "vehicle.h"
#include "tour.h"

vehicle::vehicle(std::string id, int capacity, tour tour)
	:id_{ id }, capacity_{ capacity }	//, tour_{tour}
{
}

vehicle::vehicle(nlohmann::ordered_json vehicle_json)
{
	id_ = vehicle_json.at("id_");
	//capacity_ = vehicle_json.at("capacity");	//implemented with NULL in Python
	//tour_ =

}
vehicle::~vehicle() { std::cout << "vehicle " << id_ << " destroyed"; };


//setters & getters
void vehicle::set_tour(tour tour) { tour_ = tour; };
std::string vehicle::get_id() const { return id_; }
tour vehicle::get_tour() const { return tour_; }



std::ostream& operator<<(std::ostream& os, const vehicle& vehicle)
{
	os << "Vehicle " << vehicle.id_ << ": " << "Capacity = " << vehicle.capacity_;
	return os;
}
