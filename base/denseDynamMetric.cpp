#include "denseDynamMetric.h"

/// Constructor             
/// \param dim Dimension of the target space    

denseDynamMetric::denseDynamMetric(int dim): 
dynamMetric(dim),
mB(VectorXd::Zero(mDim))
{}
