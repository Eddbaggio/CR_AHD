#pragma once
#include <string>
#include <array>

struct coordinates
{
	float m_x_coord; // convention: m_ for member variables
	float m_y_coord;

	coordinates()  // just like classes, structs can have default constructors
	{

	}

	coordinates(float x_coord, float y_coord)  // just like classes, structs can have constructors
	{
		m_x_coord = x_coord;
		m_y_coord = y_coord;
	}
};

struct time_window
{
	float m_e;
	float m_l;

	time_window()
	{
		
	}

	time_window(float open, float close)
	{
		m_e = open;
		m_l = close;
	}
};

void print(const std::string& x);
void print(int x);
void print(int* x);

const std::array<std::string, 7> univie_colors_100 = {
		"#0063A6", // universitätsblau
		"#666666", // universtitätsgrau
		"#A71C49", // weinrot
		"#DD4814", // orangerot
		"#F6A800", // goldgelb
		"#94C154", // hellgrün
		"#11897A", // mintgrün
};
