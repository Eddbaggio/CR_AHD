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
#include "utils.h"


class instance
{
private:
	std::string id_;
	std::vector<vertex> requests_;
	std::vector<carrier> carriers_;
	std::map<std::string, std::map<std::string, float>> distance_matrix_;
	bool solved_ = false;
public:
	//constructors and destructors
	instance();
	instance(const std::string& id,
	         const std::vector<vertex>& requests,
	         const std::vector<carrier>& carriers,
	         const std::map<std::string, std::map<std::string, float>>& distance_matrix);
	//move constructor
	instance(std::string&& id,
	         std::vector<vertex>&& requests,
	         std::vector<carrier>&& carriers,
	         std::map<std::string, std::map<std::string, float>>&& distance_matrix) noexcept;
	instance(const std::filesystem::path& path);
	//TODO move and copy constructors
	//TODO destructors
	~instance();

	//setters and getters
	const std::vector<carrier>& get_carriers() const;
	const std::vector<vertex>& get_requests() const;

	//operators
	friend std::ostream& operator<<(std::ostream& os, const instance& inst);

	//algorithms
	void static_I1_construction(
		std::string init_method = "earliest_due_date",
		int verbose = utils::opts.at("verbose"),
		int plot_level = utils::opts.at("plot_level")
	);
};
