#ifndef _BETA_OUTERSOFTABSMETRIC_

#include "dynamMetric.h"

///  \author Michael Betancourt
///
/// Abstract base class defining the interface
/// for a Hamiltonian defined on a Riemannian manifold
/// with an outer-product approximation to the SoftAbs Metric

class outerSoftAbsMetric: public dynamMetric
{
    
    public:
        
        explicit outerSoftAbsMetric(int dim);
        virtual ~outerSoftAbsMetric() {};
        
        //////////////////////////////////////////////////
        //                   Accessors                  //
        //////////////////////////////////////////////////
        
        double T();
        double tau();
    
        void bounceP(const VectorXd& normal);
        
        //////////////////////////////////////////////////
        //                   Mutators                   //
        //////////////////////////////////////////////////
        
        void sampleP(Random& random);
    
        /// Set the SoftAbs regularization coefficient
        void setSoftAbsAlpha(double a) { mSoftAbsAlpha = a; }
        
        //////////////////////////////////////////////////
        //              Auxiliary Functions             //
        //////////////////////////////////////////////////
        
        void checkEvolution(const double epsilon = 1e-6);
        
        void prepareEvolution() { gradV(); fComputeH();}
        
    protected:
        
        double mSoftAbsAlpha;        ///< Regualarization coefficient
    
        MatrixXd mH;                 ///< Hessian of the potential energy

        VectorXd& fLambdaDot(const VectorXd& v); ///< Compute the product of Lambda and v efficiently
        
        virtual void fComputeH() = 0; ///< Compute the Hessian matrix of the potential
    
        void fComputeMetric();
        void fPrepareSpatialGradients() { fComputeH(); gradV(); }
        
        VectorXd& dTaudp();
        VectorXd& dTaudq();
        VectorXd& dPhidq();


};

#define _BETA_OUTERSOFTABSMETRIC_
#endif