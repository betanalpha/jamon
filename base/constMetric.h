#ifndef _BETA_CONSTMETRIC_

#include "baseHamiltonian.h"

/// \author Michael Betancourt
///
/// Abstract base class defining the interface
/// for a Hamiltonian defined on a Euclidean manifold,
/// \f[ H = \frac{1}{2} p_{i} p_{j} \left(M^{-1}\right)^{ij} + V \! \left( \mathbf{q} \right) \f]
///
/// The Hamiltonian operators are defined as
/// \f[ \hat{T} = p_{i} \left(M^{-1}\right)^{ij} \frac{ \partial }{ \partial q_{j} } \f]
/// and
/// \f[ \hat{V} = - \frac{ \partial V }{ \partial q_{i} } \frac{ \partial }{ \partial p_{i} }, \f]
/// yielding the usual leapfrog evolution.

class constMetric: public baseHamiltonian
{
    
    public:
        
        explicit constMetric(int dim);
        virtual ~constMetric() {};
        
        void beginEvolveP(const double epsilon) { fEvolveP(epsilon); }
        void finishEvolveP(const double epsilon) { fEvolveP(epsilon); }
    
    protected:
    
        /// Auxiliary vector for efficient matrix computations
        VectorXd mAuxVector;
    
        /// Unified evolution for both intital and final steps
        void fEvolveP(const double epsilon);

};

#define _BETA_CONSTMETRIC_
#endif