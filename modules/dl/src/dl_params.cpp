#include "cil/dl/dl.h"

using namespace cil;
using namespace dl;


DLParams::DLParams()
{

}
DLParams::~DLParams()
{

}
DLParams::DLParams( int* architecture, int n)
{
	this->architecture = Eigen::Map<Eigen::VectorXi>(architecture,n);

}
DLParams::DLParams(Eigen::VectorXi & arc)
{
	this->architecture = arc;
}