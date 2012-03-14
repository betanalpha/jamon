#ifndef _BETA_DIAGCONSTMETRIC_

#include "constMetric.h"

///  \author Michael Betancourt
///
/// Abstract base class defining the interface
/// for a Hamiltonian with quadratic kinetic energy
/// defined on a Euclidean manifold with diagonal metric.

class diagConstMetric: public constMetric
{
    
    public:
        
        explicit diagConstMetric(int dim);
        virtual ~diagConstMetric() {};
        
        //////////////////////////////////////////////////
        //                   Accessors                  //
        //////////////////////////////////////////////////
        
        double T();
     
        void evolveQ(double epsilon);
    
        void bounceP(const VectorXd& normal);
    
        /// Return mass matrix
        VectorXd& massInv() { return mMassInv; }
        
        //////////////////////////////////////////////////
        //                   Mutators                   //
        //////////////////////////////////////////////////

        void sampleP(Random& random);
        
    protected:
        
        /// Inverse mass matrix
        VectorXd mMassInv;
    
};

#define _BETA_DIAGCONSTMETRIC_
#endif