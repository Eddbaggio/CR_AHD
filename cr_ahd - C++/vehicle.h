#pragma once

#include <string>
#include <iostream>
#include <nlohmann/json.hpp>
#include "tour.h"

class vehicle
{
private:
	std::string id_;
	int capacity_;
	tour tour_;
	//color color_;

public:
	//constructors
	vehicle(std::string id, int capacity);
	vehicle(nlohmann::ordered_json);

	// operators
	friend std::ostream& operator<<(std::ostream os, const vehicle& vehicle);

};

