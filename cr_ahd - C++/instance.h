#pragma once

#include <string>
#include <vector>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <nlohmann/json.hpp>

#include "vertex.h"
#include "vehicle.h"
#include "carrier.h"



class instance
{
private:
	std::string id_;
	std::vector<vertex> requests_;
	std::vector<carrier> carriers_;
	std::unordered_map<std::string, std::unordered_map<std::string, float>> distance_matrix_; 
public:
	//constructors and destructors
	instance();
	instance(std::string id,std::vector<vertex> requests, std::vector<carrier> carriers, std::unordered_map<std::string, std::unordered_map<std::string, float>> dist_matrix);
	instance(const std::filesystem::path& path);;
	//TODO move and copy constructors
	//TODO destructors
	~instance();

	//setters and getters
	std::vector<carrier> get_carriers() {
		return carriers_;
	}

	//operators
	friend std::ostream& operator<<(std::ostream& os, const instance& inst);

};

