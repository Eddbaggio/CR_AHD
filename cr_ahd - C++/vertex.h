#pragma once
#include <iostream>

#include "utils.h"

class vertex
{
private:
	// why do my vertices not have any id?
	coordinates coordinates_; //how to properly rename this to coordinates (same name for structure and for member variable)
	time_window time_window_; // as for coords: how to rename to time_window without error?
	float demand_{};
	float service_duration_{};
public:
	// Constructors
	vertex();
	vertex(float x, float y, float demand, float tw_open, float tw_close, float service_duration);
	~vertex();

	// Setters and Getters
	coordinates get_coordinates() const;
	void set_coordinates(coordinates coordinates);
	time_window get_time_window() const;
	void set_time_window(time_window time_window);

	// methods
	void print_coordinates();
	void dummy_func();
};
