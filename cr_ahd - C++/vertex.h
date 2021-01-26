#pragma once
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include "utils.h"

class vertex
{
private:
	std::string id_;
	coordinates coordinates_; //how to properly rename this to coordinates (same name for structure and for member variable)
	time_window time_window_; // as for coords: how to rename to time_window without error?
	float demand_{};
	float service_duration_{};

public:
	// Constructors
	vertex();
	vertex(std::string id, float x, float y, float demand, float tw_open, float tw_close, float service_duration);
	vertex(const nlohmann::json& json);
	// no destructor, not copy constructor, no copy assignment (& no move semantics) as i only use built in types as members (?!)

	// Setters and Getters
	coordinates get_coordinates() const;
	void set_coordinates(coordinates coordinates);
	time_window get_time_window() const;
	void set_time_window(time_window time_window);

	// methods
	void print_coordinates();

	//operators
	friend std::ostream& operator<<(std::ostream& os, const vertex& vertex)
	{
		os << "Vertex:\n" << "id:\t" + vertex.id_ << "\n" << "coordinates:\t" << vertex.coordinates_ << "\n" << "time window:\t" << vertex.time_window_<< "\n" << "deamnd:\t" << vertex.demand_<< "\n";
		return os;
	}
		
};
