#include <iostream>
#include "ProcessorCaps.hpp"

using namespace std;

int main(int argc, char* argv[])
{
  khyber::ProcessorCaps pcaps;
  cout << pcaps.GetCapsDescription() << endl;
  
  return 1;
}