#ifndef _BETA_DYNAMMETRIC_

#include "baseHamiltonian.h"

/// \author Michael Betancourt
///
/// Abstract base class defining the interface
/// for a Hamiltonian defined on a Riemannian manifold,
/// \f[ H = \frac{1}{2} p_{i} p_{j} \Lambda^{ij} \! \left( \mathbf{q} \right) 
/// - \frac{1}{2} \log \! \left| \mathbf{\Lambda} \! \left( \mathbf{q} \right) \right|
/// + V ! \left( \mathbf{q} \right) \f]
///
/// The Poisson operators are defined as
/// \f[ \hat{T} = p_{i} \left(\Lambda \right)^{ij} \frac{ \partial }{ \partial q_{j} } \f]
/// and
/// \f[ \hat{V} = - \frac{\partial}{\partial q_{i}} \left( \frac{1}{2} p_{i} p_{j} \Lambda^{ij} 
/// - \frac{1}{2} \log \! \left| \mathbf{\Lambda} \right| + V \right) \frac{ \partial }{ \partial p_{i} }. \f]

class dynamMetric: public baseHamiltonian
{
    
    public:
        
        explicit dynamMetric(int dim);
        virtual ~dynamMetric() {};
        
        virtual void prepareEvolution() = 0;
    
        /// Set the number of fixed point iterations for spatial updates
        void setNumFixedPoint(int n) { mNumFixedPoint = n; }
    
    protected:

        /// Number of fixed point iterations for spatial updates
        int mNumFixedPoint; 
    
        /// Auxiliary vector for efficient matrix computations
        VectorXd mAux;

        /// Evolve \f[ \hat{\mathcal{A}} = - \frac{\partial}{\partial q_{i}} 
        /// \left( \frac{1}{2} p_{i} p_{j} \Lambda^{ij} \right) 
        /// \frac{ \partial }{ \partial p_{i} }.\f]
        /// The completely time reversible implementation
        /// improves the total error by about an order of magnitude.
        virtual void fHatA(double epsilon) = 0;
    
        /// Evolve \f[ \hat{F} = - \frac{\partial}{\partial q_{i}} \left(
        /// - \frac{1}{2} \log \! \left| \mathbf{\Lambda} \right| 
        /// + V \right) \frac{ \partial }{ \partial p_{i} }\f]
        virtual void fHatF(double epsilon) = 0;
    
};

#define _BETA_DYNAMMETRIC_
#endif