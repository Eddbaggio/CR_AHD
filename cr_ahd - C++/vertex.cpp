#include "Vertex.h"

#include <iostream>

#include "utils.h"

// The class declaration goes into the header file.
// and the implementation goes in the CPP file

// default constructor; no need to type this as it will be created automatically (by the compiler?) anyways
vertex::vertex()
{

}

/**
 * \brief constructor for the Vertex Class
 * \param x x coordinate of the vertex
 * \param y y coordinate of the vertex
 * \param demand demand of vertex
 * \param tw_open earliest allowed arrival time
 * \param tw_close latest allowed arrival time
 * \param service_duration time required to fulfill service at this vertex
 */

// constructor; same name as the class; many constructors can exists with different # parameters (overloading); there is always a (potentially hidden) default constructor
vertex::vertex(
	float x,
	float y,
	float tw_open,
	float tw_close,
	float demand,
	float service_duration
)

	// member initializer list
	: coordinates_(x, y), time_window_(tw_open, tw_close), demand_(demand), service_duration_(service_duration)
{
	
}

vertex::~vertex()  // Destructor, free up memory whenever an instance is deleted
{
	print("Vertex destroyed!");
}

coordinates vertex::get_coordinates() const
{
	return coordinates_;
}

void vertex::set_coordinates(coordinates coordinates)
{
	coordinates_ = coordinates;
}

time_window vertex::get_time_window() const
{
	return time_window_;
}

void vertex::set_time_window(time_window time_window)
{
	time_window_ = time_window;
}

void vertex::print_coordinates()
{
	std::cout << coordinates_.m_x_coord << ", " << coordinates_.m_y_coord << std::endl;  // should later be replaced by overloading the << operator for the coordinates struct
}

void vertex::dummy_func()
{
	float a = coordinates_.m_y_coord + coordinates_.m_x_coord;
}
