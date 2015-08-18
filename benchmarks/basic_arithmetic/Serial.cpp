#include "Serial.hpp"

using namespace std;

void RunSerialImpl(vector<float>& dst, const vector<float>& src)
{
  for ( auto i = 0; i < dst.size(); ++i ) {
    dst[i] += src[i];
  }
}
