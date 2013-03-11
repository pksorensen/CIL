#include "cil/dl/dl.h"

using namespace cil;
using namespace dl;


DLParams::DLParams()
	: n(0)
{

}
DLParams::~DLParams()
{

}
DLParams::DLParams( int* architecture, int n)
{
	 this->architecture = Eigen::Map<Eigen::VectorXi>(architecture,n);
	 this->n = n;
}
DLParams::DLParams(Eigen::VectorXi & arc)
{
	this->architecture = arc;
	this->n = n;
}