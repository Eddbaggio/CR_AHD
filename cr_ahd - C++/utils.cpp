#include "utils.h" //including the own header is not required but increases forward compatibility: the source file has access to everything the header has access to (incl. e.g. other #includes)


void print(const std::string& x)
// using a (const std::string&) rather than just (std::string) to NOT copy the input parameter but rather just take the reference to the memory address
{
	std::cout << x << std::endl;
};

void print(int x)
{
	std::cout << x << std::endl;
}

void print(int* x)
{
	std::cout << x << std::endl;
}

float utils::travel_time(const float& distance)
{
	return (static_cast<float>(distance) / static_cast<float>(opts.at("speed_kmh"))) * static_cast<float>(std::pow(60, 2));
}
