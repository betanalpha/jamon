#ifndef _BETA_CHAINBUNDLE_

#include <vector>
#include <RandomLib/Random.hpp>

#include "baseHamiltonian.h"

using std::vector;
using RandomLib::Random;
                                           
///  \author Michael Betancourt       
///                        
///  Container for an ensemble of Markov chains,
///  including functionality for constrainted 
///  Hamiltonian Monte Carlo transitions, 
///  convergence diagnostics, and sample analysis.
///  
///  See chainBundle::evolve() for details on
///  the symplectic integrator used for the 
///  Hamiltonian evolution.

class chainBundle
{
    
    public:
        
        explicit chainBundle(Random &random);
        ~chainBundle();
        
        //////////////////////////////////////////////////
        //                   Accessors                  //
        //////////////////////////////////////////////////
        
        int nChains() { return (int)mChains.size(); }            ///< Return number of chains
        baseHamiltonian* chain(int i) { return mChains.at(i); }  ///< Return pointer to ith chain
    
        double minESS() { return mMinESS; }                      ///< Return the smallest effective sample size
        
        //////////////////////////////////////////////////
        //                   Mutators                   //
        //////////////////////////////////////////////////
        
        /// Set verbosity of transition output
        /// 0 : No output
        /// 1 : NaN warnings
        /// 2 : Above plus initial and final states along with acceptance information
        void setVerbosity(unsigned int v) { mVerbosity = v; }
        
        void setNumSubsample(int n) { mNumSubsample = n; }               ///< Set number of HMC transitions per sample
        void setNumLeapfrog(int n) { mNumLeapfrog = n; }                 ///< Set number of leapfrog steps
        void setWindowSize(int n);                                       //   Requires warning, see cpp
        
        void setStepSize(double s) { mStepSize = s; }                    ///< Set stepsize
        void setStepSizeJitter(double j);                                //   Requires warning, see cpp
        void setStepSizeJerk(double j);                                  //   Requires warning, see cpp
        void setProbStepSizeJerk(double p);                              //   Requires warning, see cpp
        
        void setNumFixedPoint(int n);                                    //   Requires warning, see cpp
        
        void setTemperAlpha(double a) { mTemperAlpha = a; }              ///< Set HMC tempering parameter
        
        void storeSamples(bool store = true) { mStoreSamples = store; }  ///< Set sample storage flag
        
        /// Add a new chain to the bundle
        void addChain(baseHamiltonian* object) { mChains.push_back(object); }
        
        //////////////////////////////////////////////////
        //              Auxiliary Functions             //
        //////////////////////////////////////////////////
        
        void evolve(baseHamiltonian* chain, double epsilon);
        void verboseEvolve(baseHamiltonian* chain, double epsilon);
    
        void transition();
        void transition(int i);
        bool transition(baseHamiltonian* chain);
    
        void checkIntError(int i, int N, int f = 1);
        
        void seed(double min, double max);
        void seed(int i, double min, double max);
        void burn(int nBurn = 100, int nCheck = 100, double minR = 1.1, bool verbose = true);
        
        void clearSamples();
        void computeSummaryStats(int b, bool verbose = true, std::ofstream* outputFile = NULL);
        
    protected:
        
        // Flags
        bool mStoreSamples;       ///< Sample storage flag
        bool mChar;               ///< Burn in completion flag
        bool mConstraint;         ///< Constraint satisfaction flag
        
        // Switches
        unsigned int mVerbosity;  ///< Verbosity level
        
        // Random number generator
        Random mRandom;           ///< Mersenne twistor pseudo-random number generator
        
        // Hamiltonian Monte Carlo parameters
        int mNumSubsample;        ///< Number of HMC transitions per sample
        int mNumLeapfrog;         ///< Number of leapfrog steps
        int mWindowSize;          ///< Size of reject/accept windows
        
        double mStepSize;         ///< Nominal leapfrog stepsize
        double mStepSizeJitter;   ///< Stepsize jitter, as fraction of current stepsize
        double mStepSizeJerk;     ///< Stepsize jerk, as fraction of current stepsize
        double mProbStepSizeJerk; ///< Probability of a stepsize jerk instead of the usual jitter
        
        int mNumFixedPoint;       ///< Number of fixed point iterations for implicit leapfrog updates
        
        double mTemperAlpha;      ///< Tempering parameter
        
        // Sample parameters
        int mNumSamples;             ///< Total number of computed samples
        vector<VectorXd> mSamples;   ///< Stored samples
        vector<double> mSampleE;     ///< Stored sample potential energies
    
        /// Smallest effective sample size, computed in computeSummaryStats()
        double mMinESS;            

        // Chain containers
        vector<baseHamiltonian*> mChains;       ///< Vector of baseChain pointers
        vector<baseHamiltonian*>::iterator mIt; ///< Vector iterator 
        
        // Private functions
        
        /// Does the chain satisfy all defined constraints?
        virtual bool fConstraint(baseHamiltonian* chain) { return !(chain->supportViolated()); }
        
        /// Return the normal to the constraint surface
        virtual const VectorXd& fNormal(baseHamiltonian* chain) { return chain->supportNormal(); }
     
        double fLogSumExp(double a, double b);
    
};

#define _BETA_CHAINBUNDLE_
#endif