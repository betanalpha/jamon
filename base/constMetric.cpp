#include "constMetric.h"

/// Constructor             
/// \param dim Dimension of the target space    

constMetric::constMetric(int dim): 
baseHamiltonian(dim),
mAux(VectorXd::Zero(mDim))
{}

void constMetric::fEvolveP(double epsilon) 
{
    mP -= epsilon * gradV();
}