#include <memory>
#include "instance.h"

instance::instance()	//default constructor
{

}

//instance::instance(std::string id, std::vector<vertex> requests, std::vector<carrier> carriers, std::unordered_map<std::string, std::unordered_map<std::string, float>> dist_matrix)
//	:id_{ id }, requests_{ requests }, carriers_{ carriers }, distance_matrix_{ dist_matrix }
//{
//}

instance::instance(const std::filesystem::path& path) // construct from json path
{
	//read json
	std::ifstream custom_instance_file_stream(path);
	nlohmann::ordered_json json_file = nlohmann::ordered_json::parse(custom_instance_file_stream);
	std::cout << "successfully read custom json instance" << '\n';

	// get id
	id_ = path.stem().string();

	// create request vertices
	for (auto& request_json : json_file.at("requests")) {
		auto r = std::make_unique<vertex>(request_json);
		requests_.push_back(std::move(r));
	}

	// create carriers
	for (auto& carrier_json : json_file.at("carriers")) {
		auto c = std::make_unique<carrier>(carrier_json);
		carriers_.push_back(std::move(c));
	}	

	//distance matrix
	std::map<std::string, std::map<std::string, float>> distance_matrix_ = json_file.at("dist_matrix");
}

instance::~instance() {
	std::cout << "instance destroyed" << '\n';
}

//setters and getters

const std::vector<std::unique_ptr<carrier>>& instance::get_carriers() {return carriers_;}

const std::vector<std::unique_ptr<vertex>>& instance::get_requests() { return requests_; }

std::ostream& operator<<(std::ostream& os, const instance& inst){
		os << "Instance:\n" << "id:\t" + inst.id_ << "\n" << "reqeusts:\t" << inst.requests_.size() << "\n" << "carriers:\t" << inst.carriers_.size() << "\n";
		return os;
}
