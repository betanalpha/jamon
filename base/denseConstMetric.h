#ifndef _BETA_DENSECONSTMETRIC_

#include <Eigen/Cholesky>

#include "constMetric.h"

///  \author Michael Betancourt
///
/// Abstract base class defining the interface
/// for a Hamiltonian with quadratic kinetic energy
/// defined on a Euclidean manifold with dense metric.

class denseConstMetric: public constMetric
{
    
    public:
        
        explicit denseConstMetric(int dim, MatrixXd& massInv);
        virtual ~denseConstMetric() {};
        
        //////////////////////////////////////////////////
        //                   Accessors                  //
        //////////////////////////////////////////////////
        
        double T();
        
        void evolveQ(double epsilon);
        
        void bounceP(const VectorXd& normal);
        
        //////////////////////////////////////////////////
        //                   Mutators                   //
        //////////////////////////////////////////////////
        
        void sampleP(Random& random);

        void setInvMass(MatrixXd& massInv);
        
    protected:
        
        /// Inverse mass matrix
        MatrixXd mMassInv;
        
        /// Cholesky factorization of mass matrix
        LLT<MatrixXd> mMassL;
    
};

#define _BETA_DENSECONSTMETRIC_
#endif