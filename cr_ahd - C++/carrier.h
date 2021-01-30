#pragma once
#include <string>
//#include <unordered_map> 
#include <map> 
#include "vertex.h"
#include "vehicle.h"
#include "nlohmann/json.hpp"

class carrier
{
private:
	std::string id_;
	vertex depot_;
	std::vector<vehicle> vehicles_;
	std::vector<vertex> requests_;
	std::vector<vertex> unrouted_; //should this be pointers to elements of unordered map requests_?
	//float cost_;
	//float revenue_;
	//float profit_;
	//int num_active_vehicles_;
	//int active_vehicles_;
	//int inactive_vehicles_;

public:
	// Constructors
	carrier();
	carrier(nlohmann::ordered_json carrier_json);
	~carrier();
	//TODO copy and move semantics!

	// Setters and Getters
	const std::vector<vertex>& get_requests() const;

	const std::vector<vehicle>& get_vehicles() const;
	
	//operators
	friend std::ostream& operator<<(std::ostream& os, carrier& carrier);

	// other member functions / methods 
	//void assign_request(const vertex& request);
	int num_vehicles() const;
	int num_requests() const;
	int num_unrouted() const;

	//algorithms
	void compute_all_vehicle_cost_and_schedules(const std::map<std::string, std::map<std::string, float>>& distance_matrix);
};

