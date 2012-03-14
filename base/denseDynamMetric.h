#ifndef _BETA_DENSEDYNAMMETRIC_

#include "dynamMetric.h"

/// \author Michael Betancourt
///
/// Abstract base class defining the interface
/// for a Hamiltonian defined on a Riemannian manifold
/// with dense metric.  
///
/// The potential Poisson operator is decomposed as
/// \f[ \hat{V} = \sum_{i} \hat{\mathcal{A}}_{i} + \hat{F}, \f]
/// where
/// \f[ \hat{\mathcal{A}}_{i} = - \frac{\partial}{\partial q_{i}} \left( \frac{1}{2} p_{i} p_{j} \Lambda^{ij} 
/// \right) \frac{ \partial }{ \partial p_{i} } \f]
/// and
/// \f[ \hat{F} = - \frac{\partial}{\partial q_{i}} \left(
/// - \frac{1}{2} \log \! \left| \mathbf{\Lambda} \right| + V \right) \frac{ \partial }{ \partial p_{i} }. \f]

class denseDynamMetric: public dynamMetric
{
    
    public:
        
        explicit denseDynamMetric(int dim);
        virtual ~denseDynamMetric() {};
    
    protected:
    
        /// Additional auxiliary vector for efficient matrix computations
        VectorXd mB; 
    
        
};

#define _BETA_DENSEDYNAMMETRIC_
#endif