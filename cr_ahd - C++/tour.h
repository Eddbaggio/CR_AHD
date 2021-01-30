#pragma once
#include <string>
#include <vector>

#include "vertex.h"
#include "utils.h"

class tour
{
private:
	std::string id_;
	std::vector<vertex> sequence_;
	std::vector<float> arrival_schedule_;
	std::vector<float> service_schedule_;
	float cost_;


public:
	//constructors, destructors, ...
	tour();
	tour(std::string id, std::vector<vertex> sequence, std::vector<float> arrival_schedule, std::vector<float> service_schedule);
	tour(const std::string& id, const std::vector<vertex>& sequence);
	~tour();

	// getters & setters
	const std::vector<vertex> get_sequence() const;

	//operators
	friend std::ostream& operator<<(std::ostream& os, const tour& t);

	//methods
	size_t size() const;
	
	//algorithms
	void compute_cost_and_schedule(
		const std::map<std::string, std::map<std::string, float>>& distance_matrix,
		const float start_time = utils::opts.at("start_time"), 
		bool ignore_tw = true,
		int verbose = utils::opts.at("verbose"));
};

