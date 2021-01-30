#include "tour.h"
#include <iostream>

tour::tour()
{
}
// constructors & destructors
tour::tour(std::string id, std::vector<vertex> sequence, std::vector<float> arrival_schedule, std::vector<float> service_schedule)
	:id_{ id }, sequence_{ sequence }, arrival_schedule_{ arrival_schedule }, service_schedule_{ service_schedule }{}

tour::tour(const std::string& id, const std::vector<vertex>& sequence)
	: id_{ id }, sequence_{ sequence }{};

tour::~tour() { std::cout << "tour " << id_ << " destroyed" << '\n'; }

// setters & getters
const std::vector<vertex> tour::get_sequence() const {return sequence_;}


//operators
std::ostream& operator<<(std::ostream& os, const tour& t)
{
	for (auto v : t.get_sequence())
		os << v.get_id() <<" -> ";
	return os;
}

// methods
size_t tour::size() const { return sequence_.size(); }

//algorithms
void tour::compute_cost_and_schedule(
	const std::map<std::string, std::map<std::string, float>>& distance_matrix,
	const float start_time,
	bool ignore_tw,
	int verbose)
{
	cost_ = 0;
	arrival_schedule_.push_back(start_time);
	service_schedule_.push_back(start_time);
	for (std::size_t rho = 1; rho < size(); ++rho) {
		auto i = sequence_[rho - 1];
		auto j = sequence_[rho];
		auto distance = distance_matrix.at(i.get_id()).at(j.get_id());
		cost_ += distance;
		auto planned_arrival = service_schedule_[rho - 1] + i.get_service_duration() + utils::travel_time(distance);
		if (verbose > 2) {
			std::cout << "Planned arrival at " << j << ": " << planned_arrival << '\n';
		}
		if (!ignore_tw) {
			assert(planned_arrival <= j.get_time_window().l_);
		}
		arrival_schedule_[rho] = planned_arrival;
	}
}

