#pragma once
#include <string>
#include <array>
#include <map>  // associative array - usually implemented as binary search trees - avg. time complexity: O(log n)
#include "vertex.h"
#include "vehicle.h"

class carrier
{
private:
	std::string id_;
	vertex depot_;
	//std::array<Vehicle, ?? > vehicles;
	int num_of_vehicles_;
	std::map<std::string, vertex> requests_;
	std::map<std::string, vertex> unrouted_;
	float cost_;
	float revenue_;
	float profit_;
	int num_active_vehicles_;
	int active_vehicles_;
	int inactive_vehicles_;

public:
	// Constructors
	carrier();
	carrier(
		const std::string& id,
		vertex depot,
		//std::array<vehicle, ? >),
		std::map<std::string, vertex> requests
	);

	// Setters and Getters
	void set_depot(vertex depot);
	vertex get_depot() const;
	void set_requests(std::map<std::string, vertex> requests);
	std::map<std::string, vertex> get_requests() const;
	void set_unrouted(std::map<std::string, vertex> unrouted);
	std::map<std::string, vertex> get_unrouted() const;
		

	// other member functions / methods
	void assign_request(vertex request);
};

