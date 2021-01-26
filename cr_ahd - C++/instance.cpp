#include "instance.h"

instance::instance()	//default constructor
{

}

instance::instance(std::string id, std::vector<vertex> requests, std::vector<carrier> carriers, std::unordered_map<std::string, std::unordered_map<std::string, float>> dist_matrix)
	:id_{ id }, requests_{ requests }, carriers_{ carriers }, distance_matrix_{ dist_matrix }
{
}

instance::instance(const std::filesystem::path& path) // construct from json path
{
	//read json
	std::ifstream custom_instance_file_stream(path);
	nlohmann::json json_file = nlohmann::json::parse(custom_instance_file_stream);
	std::cout << "successfully read custom json instance" << '\n';

	// get id
	id_ = path.stem().string();

	// create request vertices
	std::vector<vertex> requests_;	//initialize empty
	for (auto& request_json : json_file.at("requests")) {
		requests_.push_back(vertex(request_json));
	}

	// create carriers
	std::vector<carrier> carriers_; //initialize empty
	for (auto& carrier_json : json_file.at("carriers")) {
		carriers_.push_back(carrier(carrier_json));
	}

	//distance matrix
	//TODO should this rather be a vector than an unordered map? But then, i cannot access by column name (e.g. "r0", "d1") but am forced to use integer indices
	std::unordered_map<std::string, std::unordered_map<std::string, float>> distance_matrix_;
	for (auto& [key, value] : json_file.at("dist_matrix").items()) {
		//std::unordered_map<std::string, float> value_umap;
		distance_matrix_.insert(std::pair<std::string, std::unordered_map<std::string, float>>(key, value));
	}
}

instance::~instance() {
	std::cout << "instance destroyed" << '\n';
}

std::ostream& operator<<(std::ostream& os, const instance& inst)
{
		os << "Instance:\n" << "id:\t" + inst.id_ << "\n" << "reqeusts:\t" << inst.requests_.size() << "\n" << "carriers:\t" << inst.carriers_.size() << "\n";
		return os;
}
