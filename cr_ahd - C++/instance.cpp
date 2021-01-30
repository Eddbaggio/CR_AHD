#include <memory>
#include "instance.h"

instance::instance() //default constructor
{
}

instance::instance(const std::string& id,
                   const std::vector<vertex>& requests,
                   const std::vector<carrier>& carriers,
                   const std::map<std::string, std::map<std::string, float>>& distance_matrix)
	: id_(id), requests_(requests), carriers_(carriers), distance_matrix_(distance_matrix)
{
}

//move constructor
instance::instance(std::string&& id,
                   std::vector<vertex>&& requests,
                   std::vector<carrier>&& carriers,
                   std::map<std::string, std::map<std::string, float>>&& distance_matrix) noexcept
{
}


instance::instance(const std::filesystem::path& path) // construct from json path
{
	//read json
	std::ifstream custom_instance_file_stream(path);
	nlohmann::ordered_json json_file = nlohmann::ordered_json::parse(custom_instance_file_stream);
	std::cout << "successfully read custom json instance" << '\n';

	// get id
	id_ = path.stem().string();

	// create request vertices
	for (auto& request_json : json_file.at("requests"))
	{
		auto r = vertex(request_json);
		requests_.push_back(r);
	}

	// create carriers
	for (auto& carrier_json : json_file.at("carriers"))
	{
		auto c = carrier(carrier_json);
		carriers_.push_back(c);
	}

	//distance matrix
	distance_matrix_ = json_file.at("dist_matrix").get<std::map<std::string, std::map<std::string, float>>>();	// does not work, don't know why not
}

instance::~instance()
{
	std::cout << "instance destroyed" << '\n';
}

//setters and getters
const std::vector<carrier>& instance::get_carriers() const { return carriers_; }
const std::vector<vertex>& instance::get_requests() const { return requests_; }

//operators
std::ostream& operator<<(std::ostream& os, const instance& inst)
{
	os << "Instance:\n" << "id:\t" + inst.id_ << "\n" << "requests:\t" << inst.requests_.size() << "\n" << "carriers:\t"
		<< inst.carriers_.size() << "\n";
	return os;
}

//algorithms
void instance::static_I1_construction(std::string init_method, int verbose, int plot_level)
{
	assert(solved_ == false);
	if (verbose > 0)
		std::cout << "STATIC I1 construction for " << id_ << '\n';
	//timer timer;
	//timer.start()

	for (auto& c : carriers_)
	{
		//if (plot_level>1){}
		c.compute_all_vehicle_cost_and_schedules(distance_matrix_);
	}
}
