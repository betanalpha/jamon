#ifndef _BETA_DYNAMMETRIC_

#include "baseHamiltonian.h"

/// \author Michael Betancourt
///
/// Abstract base class defining the interface
/// for a Hamiltonian defined on a Riemannian manifold,
/// \f$ H = \tau + \phi \f$, where
/// \f[ \tau = \frac{1}{2} p_{i} p_{j} \Lambda^{ij} \! \left( \mathbf{q} \right) \f]
/// and
/// \f[ \frac{1}{2} \log \! \left| \mathbf{\Sigma} \! \left( \mathbf{q} \right) \right|
/// + V ! \left( \mathbf{q} \right). \f]
///
/// The Poisson operators are defined as 
/// \f[ \hat{H} = \frac{1}{2} \hat{\phi} + \frac{1}{2} \hat{\tau} + \hat{T} + \frac{1}{2} \hat{\tau} + \frac{1}{2} \hat{\phi}, \f]
/// where
/// \f[ \hat{\phi} = \frac{\partial}{\partial q_{i}} \left(
/// \frac{1}{2} \log \! \left| \mathbf{\Sigma} \right| 
/// + V \right) \frac{ \partial }{ \partial p_{i} }, \f]
/// \f[ \hat{\tau} = - \frac{\partial}{\partial q_{k}} 
/// \left( \frac{1}{2} p_{i} p_{j} \Lambda^{ij} \right) 
/// \frac{ \partial }{ \partial p_{k} },\f]
/// and
/// \f[ \hat{T} = p_{i} \left(\Lambda \right)^{ij} \frac{ \partial }{ \partial q_{j} }. \f]
///

class dynamMetric: public baseHamiltonian
{
    
    public:
        
        explicit dynamMetric(int dim);
        virtual ~dynamMetric() {};
    
        //////////////////////////////////////////////////
        //                   Accessors                  //
        //////////////////////////////////////////////////
        
        /// Return the quadratic form of the kinetic energy
        virtual double tau() = 0;
    
        /// Return the pseudo-potential
        double phi() { return 0.5 * mLogDetMetric + V(); }
        
        //////////////////////////////////////////////////
        //                   Mutators                   //
        //////////////////////////////////////////////////
        
        /// Set the maximum number of fixed-point iterations for implicit updates
        void setMaxNumFixedPoint(int n) { mMaxNumFixedPoint = n; }
    
        /// Set the difference threshold for terminating fixed-point iterations
        void setFixedPointThreshold(double t) { mFixedPointThreshold = t; }
        
        void evolveQ(const double epsilon);
        void beginEvolveP(const double epsilon);
        void finishEvolveP(const double epsilon);  
    
        //////////////////////////////////////////////////
        //              Auxiliary Functions             //
        //////////////////////////////////////////////////

        virtual void prepareEvolution() { fComputeMetric(); fPrepareSpatialGradients(); }
    
    protected:

        /// Maximum number of fixed-point iterations for implicit updates
        int mMaxNumFixedPoint; 
    
        /// Difference threshold for terminating fixed-point iterations
        double mFixedPointThreshold;
    
        /// Auxiliary vector for storing various gradients
        VectorXd mAuxVector;
    
        /// Auxiliary vector for fixed-point iterations
        VectorXd mInit; 
    
        /// Auxiliary vector for fixed-point iteration termination criteria
        VectorXd mDelta;
    
        /// Log determinant of the metric
        double mLogDetMetric;

        /// Compute the metric at the current position
        virtual void fComputeMetric() = 0;
    
        /// Update any momenta-dependent auxiliary terms
        virtual void fUpdateP() {};
    
        /// Update any position-dependent auxiliary terms
        /// in preparation for spatial gradients
        virtual void fPrepareSpatialGradients() { gradV(); }
        
        /// Evolve \f[ \hat{T} = p_{i} \Lambda^{ij} \frac{ \partial }{ \partial q_{j} }\f]
        virtual void fHatT(const double epsilon);
    
        /// Evolve \f[ \hat{\tau} = - \frac{\partial}{\partial q_{k}} 
        /// \left( \frac{1}{2} p_{i} p_{j} \Lambda^{ij} \right) 
        /// \frac{ \partial }{ \partial p_{k} }.\f]
        virtual void fHatTau(const double epsilon, const int numFixedPoint);
    
        /// Evolve \f[ \hat{\phi} = \frac{\partial}{\partial q_{i}} \left(
        /// \frac{1}{2} \log \! \left| \mathbf{\Sigma} \right| 
        /// + V \right) \frac{ \partial }{ \partial p_{i} }\f]
        void fHatPhi(const double epsilon);
    
        /// Gradient of the quadrtic form with respect to the momenta
        virtual VectorXd& dTaudp() = 0;
    
        /// Gradient of the quadratic form with respect to the position
        virtual VectorXd& dTaudq() = 0;
    
        /// Gradient of the psuedo-potential with respect to the momenta
        virtual VectorXd& dPhidq() = 0;
    
};

#define _BETA_DYNAMMETRIC_
#endif