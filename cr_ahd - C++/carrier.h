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
	std::vector<std::unique_ptr<vehicle>> vehicles_;
	std::vector<std::unique_ptr<vertex>> requests_;
	std::vector<std::unique_ptr<vertex>> unrouted_; //should this be pointers to elements of unordered map requests_?
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
	const std::vector<std::unique_ptr<vertex>>& get_requests();
	
	//operators
	friend std::ostream& operator<<(std::ostream& os, carrier& carrier);

	// other member functions / methods
	//void assign_request(const vertex& request);
	int num_vehicles();
	int num_requests();
	int num_unrouted();
};

