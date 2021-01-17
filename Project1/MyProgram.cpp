import Foo;
//import std.core;
#include <iostream>

using namespace std;

int main()
{
	cout << "The result of f() is " << Bar::f() << endl; // 42
	 int i = Bar::f_internal(); // C2039
	 int j = ANSWER; //C2065
}