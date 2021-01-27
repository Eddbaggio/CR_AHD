#include <iostream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <fstream>

#include "instance.h"
#include "vertex.h"

using path = std::filesystem::path;

int main() {
	path json_path = path("../data/Input/Custom/C101/C101_3_15_ass_#001.json");
	

	instance instance{json_path};
	std::cout << instance;

	for (auto const& r : instance.get_requests())
		std::cout << *r << '\n';

	for (auto const& c : instance.get_carriers()) {
		std::cout << *c << '\n';
		for (auto const& r : c->get_requests()) {
			std::cout << *r << '\n';
		}
	}


}