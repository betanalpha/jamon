#include "constMetric.h"

/// Constructor             
/// \param dim Dimension of the target space    

constMetric::constMetric(int dim): 
baseHamiltonian(dim),
mAuxVector(VectorXd::Zero(mDim))
{}

void constMetric::fEvolveP(const double epsilon) 
{
    mP -= epsilon * gradV();
}