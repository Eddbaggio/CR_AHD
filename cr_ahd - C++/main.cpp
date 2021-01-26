#include <iostream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <fstream>

#include "instance.h"
#include "vertex.h"

using json = nlohmann::json;
using path = std::filesystem::path;

int main() {
	path json_path = path("../data/Input/Custom/C101/C101_3_15_ass_#001.json");
	
	//std::cout << custom_instance_json.dump(4) << "\n";

	instance instance{json_path};
	for (auto& carrier:instance.get_carriers())

	std::cout << instance;

}