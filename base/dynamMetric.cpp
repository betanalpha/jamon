#include <iostream>
#include <iomanip>

#include "dynamMetric.h"

/// Constructor             
/// \param dim Dimension of the target space    

dynamMetric::dynamMetric(int dim): 
baseHamiltonian(dim),
mAux(VectorXd::Zero(mDim)),
mNumFixedPoint(5)
{}