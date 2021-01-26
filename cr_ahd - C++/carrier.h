#pragma once
#include <string>
#include <unordered_map> 
#include "vertex.h"
#include "vehicle.h"
#include "nlohmann/json.hpp"

class carrier
{
private:
	std::string id_;
	vertex depot_;
	std::vector<vehicle> vehicles_;
	int num_of_vehicles_;
	std::unordered_map<std::string, vertex> requests_;
	//std::unordered_map<std::string, vertex> unrouted_; //should be pointers to elements of unordered map requests_
	float cost_;
	float revenue_;
	float profit_;
	int num_active_vehicles_;
	int active_vehicles_;
	int inactive_vehicles_;

public:
	// Constructors
	carrier();
	//carrier(
	//	const std::string& id,
	//	vertex depot,
	//	std::vector<vehicle>,
	//	std::map<std::string, vertex> requests

	//);
	carrier(nlohmann::json carrier_json);

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

