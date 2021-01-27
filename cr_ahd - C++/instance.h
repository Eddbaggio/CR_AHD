#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
//#include <unordered_map>
#include <map>
#include <vector>

#include <nlohmann/json.hpp>

#include "vertex.h"
#include "vehicle.h"
#include "carrier.h"



class instance
{
private:
	std::string id_;
	std::vector<std::unique_ptr<vertex>> requests_;
	std::vector<std::unique_ptr<carrier>> carriers_;
	std::map<std::string, std::map<std::string, float>> distance_matrix_; 
public:
	//constructors and destructors
	instance();
	//instance(std::string id,std::vector<vertex> requests, std::vector<carrier> carriers, std::unordered_map<std::string, std::unordered_map<std::string, float>> dist_matrix);
	instance(const std::filesystem::path& path);;
	//TODO move and copy constructors
	//TODO destructors
	~instance();

	//setters and getters
	const std::vector<std::unique_ptr<carrier>>& get_carriers();
	const std::vector<std::unique_ptr<vertex>>& get_requests();

	//operators
	friend std::ostream& operator<<(std::ostream& os, const instance& inst);

};

