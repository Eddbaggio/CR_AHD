#pragma once

#include <string>
#include <iostream>
#include <nlohmann/json.hpp>
#include <memory>

#include "tour.h"

class vehicle
{
private:
	std::string id_;
	int capacity_;
	tour tour_;
	//color color_;

public:
	//constructors & destructors
	vehicle(std::string id, int capacity, tour tour);
	vehicle(nlohmann::ordered_json);
	~vehicle();

	//setters & getters
	void set_tour(tour tour);
	std::string get_id() const;
	tour get_tour() const;

	// operators
	friend std::ostream& operator<<(std::ostream& os, const vehicle& vehicle);

	//methods
};

