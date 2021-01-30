#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <cmath>

struct coordinates
{
	float x_coordinate_; // convention: m_ for member variables
	float y_coordinate_;

	coordinates()  // just like classes, structs can have default constructors
	{

	}

	coordinates(float x_coord, float y_coord)  // just like classes, structs can have constructors
	{
		x_coordinate_ = x_coord;
		y_coordinate_ = y_coord;
	}

	friend std::ostream& operator<<(std::ostream& os, const coordinates& coordinates)
	{
		os << "(" << coordinates.x_coordinate_ << ", " << coordinates.y_coordinate_ << ")";
		return os;
	}

};

struct time_window
{
	float e_;
	float l_;

	time_window()
	{
		
	}

	time_window(float open, float close)
	{
		e_ = open;
		l_ = close;
	}

	friend std::ostream& operator<<(std::ostream& os, const time_window& time_window)
	{
		os << "(" << time_window.e_ << ", " << time_window.l_<< ")";
		return os;
	}
};


namespace utils {
	//const std::vector<std::string> univie_colors_100 = {
	//		"#0063A6", // universitätsblau
	//		"#666666", // universtitätsgrau
	//		"#A71C49", // weinrot
	//		"#DD4814", // orangerot
	//		"#F6A800", // goldgelb
	//		"#94C154", // hellgrün
	//		"#11897A", // mintgrün
	//};

	
	const std::unordered_map<std::string, float> opts = {
		{"verbose", static_cast<float>(3.0)},
		//{"plot_level", 1.0},
		//{"speed_kmh", static_cast<float>(std::pow(60, 2))},
		{"start_time", static_cast < float>(0.0)},
		//{"alpha_1", 0.5},
		//{"mu", 1.0},
		//{"lambda", 2.0},
	};

	float travel_time(const float& distance);
}
