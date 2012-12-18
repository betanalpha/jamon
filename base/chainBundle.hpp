#ifndef _BETA_CHAINBUNDLE_

#include <vector>
#include <RandomLib/Random.hpp>

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

template <typename T>
class chainBundle
{
    
    public:
        
        explicit chainBundle(Random &random);
        ~chainBundle();
        
        //////////////////////////////////////////////////
        //                   Accessors                  //
        //////////////////////////////////////////////////
        
        int nChains() { return (int)mChains.size(); }                  ///< Return number of chains
        T* chain(const int i) { return mChains.at(i); }                ///< Return pointer to ith chain
    
        double stepSize() { return mStepSize; }                        ///< Return current stepsize
        double nLeapfrog() { return mNumLeapfrog; }                    ///< Return current number of leapfrog steps
    
        double ESS(int i) { return mESS(i); }                          ///< Return the ith effective sample size
        double minESS() { return mESS.minCoeff(); }                    ///< Return the smallest effective sample size
        
        vector<VectorXd>& samples() { return mSamples; }               ///< Return accumulated samples
    
        //////////////////////////////////////////////////
        //                   Mutators                   //
        //////////////////////////////////////////////////
        
        /// Set verbosity of transition output
        /// 0 : No output
        /// 1 : NaN warnings
        /// 2 : Above plus initial and final states along with acceptance information
        void setVerbosity(const unsigned int v) { mVerbosity = v; }
        
        void setNumSubsample(const int n) { mNumSubsample = n; }        ///< Set number of HMC transitions per sample
        void setNumLeapfrog(const int n) { mNumLeapfrog = n; }          ///< Set number of leapfrog steps
        void setWindowSize(const int n);                                //   Requires warning, see cpp
        
        void setStepSize(const double s) { mStepSize = s; }             ///< Set stepsize
        void setStepSizeJitter(const double j);                         //   Requires warning, see cpp
        void setStepSizeJerk(const double j);                           //   Requires warning, see cpp
        void setProbStepSizeJerk(const double p);                       //   Requires warning, see cpp

        void setTemperAlpha(const double a) { mTemperAlpha = a; }       ///< Set HMC tempering parameter
        
        void setAdaptIntTime(double t) { mAdaptIntTime = t; }           ///< Set target integration time for stepsize adaptation
        void setAdaptTargetAccept(double a) { mAdaptTargetAccept = a; } ///< Set target accept rate for stepsize adaptation
        void setAdaptMaxLeapfrog(int n) { mAdaptMaxLeapfrog = n; }      ///< Set maximum number of leapfrog steps for stepsize adaptation
        
        void storeSamples(const bool store = true) { mStoreSamples = store; }  ///< Set sample storage flag
        
        void setMaxLagDisplay(int n) { mMaxLagDisplay = n; } ///< Set maximum displayed lag
        void setMaxLagCalc(int n) { mMaxLagCalc = n; }       ///< Set maximum calculated lag
                              
        void useExternalMoments(VectorXd& mean, VectorXd& var) { mUseExternalMoments = true; mExternalMean = &mean; mExternalVar = &var; }
    
        /// Add a new chain to the bundle
        void addChain(T* object) { mChains.push_back(object); }
        
        //////////////////////////////////////////////////
        //              Auxiliary Functions             //
        //////////////////////////////////////////////////
        
        void evolve(T* chain, const double epsilon);
        void verboseEvolve(T* chain, const double epsilon);

        void engageAdaptation() { mAdaptStepSize = true; }     ///< Engage dual-averaging stepsize adaptation
        void disengageAdaptation() { mAdaptStepSize = false; } ///< Disengage dual-averaging stepsize adaptation
        
        void initAdaptation();
        void updateAdaptation(double metropolisProb);
    
        void transition();
        void transition(int i);
        bool transition(T* chain);
    
        void saveTrajectory(std::ofstream& outputFile, const int i = 0);
    
        void checkIntError(const int i, const int N, const int f = 1);
        
        void seed(const double min, const double max);
        void seed(const int i, const double min, const double max);
        void burn(const int nBurn = 100, const int nCheck = 100, const double minR = 1.1, const bool verbose = true);
        
        void clearSamples();
        void computeSummaryStats(const int b, const bool verbose = true, std::ofstream* outputFile = NULL);
        
    protected:
        
        // Flags
        bool mStoreSamples;       ///< Sample storage flag
        bool mAdaptStepSize;      ///< Step-size adaptation flag
        bool mRefStats;           ///< Reference statistics flag
        bool mChar;               ///< Burn in completion flag
        bool mConstraint;         ///< Constraint satisfaction flag
        bool mUseExternalMoments; ///< External moments for autocorrelation calculation flag
        
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
    
        double mAdaptTargetAccept; ///< Stepsize adaptation target accept probability
        double mAdaptIntTime;      ///< Stepsize adaptation target integration time
        int mAdaptMaxLeapfrog;     ///< Stepsize adaptation maximum number of leapfrog steps
        double mAdaptMu;           ///< Stepsize adaptation mu
        double mAdaptLogEpsBar;    ///< Stepsize adaptation epsilon average
        double mAdaptHBar;         ///< Stepsize adaptation H average
        double mAdaptCounter;      ///< Stepsize adaptation counter
        double mAdaptGamma;        ///< Stepsize adaptation shrinkage parameter
        double mAdaptKappa;        ///< Stepsize adaptation decay parameter
        double mAdaptT0;           ///< Stepsize adaptation initialization parameter
        
        // Sample parameters
        int mNumSamples;             ///< Total number of computed samples
        vector<VectorXd> mSamples;   ///< Stored samples
        vector<double> mSampleE;     ///< Stored sample potential energies

        int mMaxLagDisplay;          ///< Maximum lag to display/save
        int mMaxLagCalc;             ///< Maximum lag used in calculation of autocorrelations
    
        /// Vector of effective sample sizes, computed in computeSummaryStats()
        VectorXd mESS;           
    
        VectorXd* mExternalMean;     ///< External mean for calculating the autocorrelations
        VectorXd* mExternalVar;      ///< External variance for calculating the autocorrelations

        // Chain containers
        vector<T*> mChains;       ///< Vector of baseChain pointers
        typename vector<T*>::iterator mIt; ///< Vector iterator 
        
        // Private functions
    
        /// Does the chain satisfy all defined constraints?
        virtual bool fConstraint(T* chain) { return !(chain->supportViolated()); }
        
        /// Return the normal to the constraint surface
        virtual const VectorXd& fNormal(T* chain) { return chain->supportNormal(); }
     
        double fLogSumExp(double a, double b);
    
};

// C++ standard libraries
#include "math.h"
#include <iostream>
#include <iomanip>
#include <fstream>

using std::cout;
using std::endl;
using std::flush;

/// Constructor
/// \param random Pointer to externally instantiated random number generator
/// \see ~chainBundle()
template <typename T>
chainBundle<T>::chainBundle(Random& random): 
mRandom(random),
mESS(VectorXd::Zero(1))
{
    
    mStoreSamples = true;
    mAdaptStepSize = false;
    mRefStats = false;
    mChar = false;
    mConstraint = true;
    mUseExternalMoments = false;
    
    mVerbosity = 0;
    
    mNumSubsample = 1;
    mNumLeapfrog = 150;
    mWindowSize = 10;
    
    mStepSize = 0.1;
    mStepSizeJitter = 0.05;
    mStepSizeJerk = 0.1;
    mProbStepSizeJerk = 0.0;
    
    mTemperAlpha = 0.0;
    
    mAdaptTargetAccept = 0.65;
    mAdaptIntTime = 10;
    mAdaptMaxLeapfrog = 1000;
    mAdaptMu = 0;
    mAdaptLogEpsBar = 0;
    mAdaptHBar = 0;
    mAdaptCounter = 0;
    mAdaptGamma = 0.05;
    mAdaptKappa = 0.75;
    mAdaptT0 = 10;   
    
    mNumSamples = 0;
    
    mMaxLagDisplay = 10;
    mMaxLagCalc = 50;

    mExternalMean = 0;
    mExternalVar = 0;
    
}


/// Destructor
/// \see chainBundle()

template <typename T>
chainBundle<T>::~chainBundle()
{
    for(mIt = mChains.begin(); mIt != mChains.end(); ++mIt) delete *mIt;
    mChains.clear();
    
    clearSamples();
    
}

/// Set reject/accept window size

template <typename T>
void chainBundle<T>::setWindowSize(const int n)
{
    
    if(n > 0.5 * mNumLeapfrog)
    {
        cout << "chainBundle::setWindowSize() - Reject/Accept window must not be larger"
        << " than half of the total leapfrog steps!" << endl;
        return;
    }
    
    if(n < 2)
    {
        //cout << "chainBundle::setWindowSize() - Reject/Accept window should be larger than two"
        //     << " to avoid proposed windows in violation of the defined constraints!" << endl; 
        //return;
    }
    
    mWindowSize = n;
    
}

/// Set valid step size jitter, as a fraction of the stepsize      

template <typename T>
void chainBundle<T>::setStepSizeJitter(const double j)
{
    
    if(j < 0 || j > 1)
    {
        cout << "chainBundle::setStepSizeJitter() - Step size jitter must lie between 0 and 1!" << endl;
        return;
    }
    
    mStepSizeJitter = j;
    
}

/// Set valid step size jerk, as a fraction of the stepsize    

template <typename T>
void chainBundle<T>::setStepSizeJerk(const double j)
{
    
    if(j < 0)
    {
        cout << "chainBundle::setStepSizeJerk() - Step size jerk cannot be negative!" << endl;
        return;
    }
    
    mStepSizeJitter = j;
    
}

/// Set valid step size jitter, as a fraction of the stepsize 

template <typename T>
void chainBundle<T>::setProbStepSizeJerk(const double p)
{
    
    if(p < 0 || p > 1)
    {
        cout << "chainBundle::setProbStepSizeJerk() - Step size jerk probability must lie between 0 and 1!" << endl;
        return;
    }
    
    mProbStepSizeJerk = p;
    
}

/// Clear all stored samples

template <typename T>
void chainBundle<T>::clearSamples()
{
    mSamples.clear();
    mSampleE.clear();
    for(mIt = mChains.begin(); mIt != mChains.end(); ++mIt) (*mIt)->clearHistory();
    
}

/// Compute summary statistics for the accumulated samples,
/// averaged into batches of size b.  
/// If a pointer to an output file is passed then the file is filled
/// with batch and autocorrelation information ready for parsing
/// with the included gnuplot script, plotTrace.
///
/// Note that the Monte Carlo Standard Error assumes independent batches.
/// 
/// /param ref Index of chain whose mean and variance will be used in computing the autocorrelations
/// /param b Number of samples to be averaged into a single batch
/// /param verbose Flag for verbose diagnostic output
/// /param outputFile Pointer to an external ofstream instance for optional text output

template <typename T>
void chainBundle<T>::computeSummaryStats(const int b, const bool verbose, std::ofstream* outputFile)
{
    
    cout.precision(6);
    int acceptWidth = 14;
    
    if(mSamples.size() == 0)
    {
        cout << "Cannot compute summary stats without any samples!" << endl;
        return;
    }
    
    if(verbose)
    {
        
        cout << "Displaying summary statistics..." << endl << endl;
        
        cout << "    " << std::setw(5 * acceptWidth / 2) << std::setfill('-') << "" << std::setfill(' ') << endl;
        cout << "    "
             << std::setw(acceptWidth / 2) << std::left << "Chain" 
             << std::setw(acceptWidth) << std::left << "Global"
             << std::setw(acceptWidth) << std::left << "Local"
             << std::endl;
        cout << "    "
             << std::setw(acceptWidth / 2) << std::left << "" 
             << std::setw(acceptWidth) << std::left << "Accept Rate"
             << std::setw(acceptWidth) << std::left << "Accept Rate"
             << std::endl;
        cout << "    " << std::setw(5 * acceptWidth / 2) << std::setfill('-') << "" << std::setfill(' ') << endl;
        
        for(int i = 0; i < mChains.size(); ++i)
        {
            cout << "    "
                 << std::setw(acceptWidth / 2) << std::left << i 
                 << std::setw(acceptWidth) << std::left << (mChains.at(i))->acceptRate()
                 << std::setw(acceptWidth) << std::left << (mChains.at(i))->movingAcceptRate()
                 << endl;
        }
        cout << "    " << std::setw(5 * acceptWidth / 2) << std::setfill('-') << "" << std::setfill(' ') << endl;
        cout << endl;
        
    }
    
    int nSamples = (int)mSamples.size();
    int nBatch = nSamples / b;
    int nLag = nBatch / 2;
    
    nLag = nLag > mMaxLagCalc ? mMaxLagDisplay : nLag;
    int nLagDisplay = nLag > mMaxLagDisplay ? mMaxLagDisplay : nLag;
    
    if(verbose)
    {
        cout << "\tUsing " << nBatch << " batches of " << b << " samples"
        << " (" << nSamples % b << " samples clipped)" << endl << endl;
    }
    
    // Loop over samples
    double batchPotential;
    double sumPotential = 0;
    double sumPotential2 = 0;
    
    long dim = chain(0)->dim();
    if(outputFile) *outputFile << "DIM " << dim << endl << endl;
    
    VectorXd batchSample(dim);
    VectorXd sum(VectorXd::Zero(dim));
    VectorXd sum2(VectorXd::Zero(dim));
    
    MatrixXd autoCorr = MatrixXd::Zero(dim, nLag);
    mESS.resize(dim);
    
    vector<VectorXd> batchSamples;
    
    vector<double>::const_iterator eIt = mSampleE.begin();
    vector<VectorXd>::const_iterator it = mSamples.begin();
    
    // Compute batch expectations
    if(outputFile) *outputFile << "BEGINSAMPLES" << endl;
    
    for(int n = 0; n < nBatch; ++n)
    {
        
        batchPotential = 0;
        batchSample.setZero();
        
        // Loop over intra-batch samples
        for(int m = 0; m < b; ++m, ++eIt, ++it)
        {
            
            // Increment sums
            batchPotential += *eIt;
            batchSample += (*it);
            
            // Write to output file, if desired
            if(outputFile)
            {
                *outputFile << n * b + m << "\t" << *eIt << flush;
                for(int i = 0; i < dim; ++i)
                {
                    *outputFile << "\t" << (*it)(i) << flush;
                }
                *outputFile << endl;
            }
        }
        
        // Normalize
        batchPotential /= b;
        sumPotential += batchPotential;
        sumPotential2 += batchPotential * batchPotential;
        
        batchSample /= (double)b;
        batchSamples.push_back(VectorXd(batchSample));
        
        sum += batchSample;
        sum2 += batchSample.array().square().matrix();
        
    }
    
    if(outputFile) *outputFile << "ENDSAMPLES" << endl << endl;
    
    // Normalize batch expectations
    sumPotential /= (double)nBatch;
    sumPotential2 = sumPotential2 / (double)nBatch - sumPotential * sumPotential;
    
    sum /= (double)nBatch;
    
    sum2 /= (double)nBatch; 
    sum2 -= sum.array().square().matrix();
    sum2 *= b / nSamples; // Correct asymptotic batch variance to asymptotic chain variance
    
    // Compute autocorrelations
    VectorXd gamma(dim);
    VectorXd one(dim);
    VectorXd two(dim);
    
    if(!mUseExternalMoments) mExternalMean = &sum;
    
    for(int k = 0; k < nLag; ++k)
    {
        
        gamma = autoCorr.col(k);
        
        vector<VectorXd>::iterator itOne = batchSamples.begin();
        
        vector<VectorXd>::iterator itTwo = itOne;
        for(int j = 0; j < k; ++j) ++itTwo;
        
        for(int n = 0; n < nBatch - k; ++n, ++itOne, ++itTwo)
        {
            one = *itOne - *mExternalMean;
            two = *itTwo - *mExternalMean;
            gamma += one.cwiseProduct(two);
        }
        
        gamma /= (double)nBatch;
        
        // Compute autocorrelation
        if(k == 0) 
        {
            for(int i = 0; i < dim; ++i) autoCorr(i, k) = gamma(i);
            continue;
        }
        
        one = gamma.cwiseQuotient(autoCorr.col(0));
        
        if(mUseExternalMoments) one = gamma.cwiseQuotient(*mExternalVar);
        else                    one = gamma.cwiseQuotient(autoCorr.col(0));
     
        for(int i = 0; i < dim; ++i) autoCorr(i, k) = one(i);
        
    }
    
    autoCorr.col(0).setOnes();
    
    // Compute effective sample size for each variable
    for(int i = 0; i < dim; ++i)
    {
        
        mESS(i) = 0;
        
        double gamma = 0;
        double Gamma = 0;
        double oldGamma = 1;
        
        for(int k = 1; k < nLag; ++k)
        {
            
            gamma = autoCorr(i, k);
            
            // Even k, finish computation of Gamma
            if(k % 2 == 0)
            {
                
                Gamma += gamma;
                
                // Convert to initial monotone sequence (IMS)
                if(Gamma > oldGamma) Gamma = oldGamma;
                
                oldGamma = Gamma;
                
                // Terminate if necessary
                if(Gamma <= 0) break;
                
                // Increment autocorrelation sum
                mESS(i) += Gamma;
                
            }
            // Odd k, begin computation of Gamma
            else
            {
                Gamma = gamma;   
            }
            
            //ess(i) += (1.0 - (double)k / (double)nBatch) * (*(autoCorr[k]))[i];
            
        }
        
        mESS(i) = (double)nBatch / (1.0 + 2 * mESS(i) );
        
    }
    
    // Display results
    if(verbose)
    {
        cout << "\tV ~ " << sumPotential << " +/- " << sqrt(sumPotential2) << endl;
        cout << "\tSmallest Effective Sample Size is " << minESS() << endl << endl;
    }
    
    double whiteNoise = 2.0 / sqrt((double)nBatch);
    
    if(outputFile) 
    {
        
        *outputFile << "WHITENOISE " << whiteNoise << endl << endl;
        *outputFile << "BEGINAUTOCORR" << endl;
        
        for(int k = 0; k < nLag; ++k)
        {
            
            *outputFile << k << flush;
            
            for(int i = 0; i < dim; ++i)
            {
                *outputFile << "\t" << autoCorr(i, k) << flush;
            }
            
            *outputFile << endl;
            
        }
        
        *outputFile << "ENDAUTOCORR" << endl;
        
    }
    
    if(verbose)
    {
        
        for(int i = 0; i < dim; ++i)
        {
            
            cout << "\t" << std::setw(4 * acceptWidth) << std::setfill('-') << "" << std::setfill(' ') << endl;
            cout << "\tVariable " << i << endl;
            cout << "\t" << std::setw(4 * acceptWidth) << std::setfill('-') << "" << std::setfill(' ') << endl;
            cout << endl;
            
            cout << "\t\tMean = " << sum[i] << endl;
            cout << "\t\tMonte Carlo Standard Error = " << sqrt(sum2[i]) << endl;
            cout << "\t\tEffective Sample Size = " << mESS(i) << endl << endl;
            
            cout << endl << "\t\tAutocorrelation:" << endl;
            cout << "\t\t\tWhite noise correlation = " << whiteNoise << endl << endl;
            
            
            for(int k = 0; k < nLagDisplay; ++k)
            {
                double g = autoCorr(i, k);
                cout << "\t\t\tgamma_{" << k << "} = " << g << flush;
                if(g > whiteNoise && k > 0) cout << " (Larger than expected fluctuation from white noise!)" << flush;
                cout << endl;
            }
            
            cout << endl;
            
        }
        
        cout << "\t" << std::setw(4 * acceptWidth) << std::setfill('-') << "" << std::setfill(' ') << endl;
        
    }
    
    // Clean up
    batchSamples.clear();
    
    return;
    
}

/// Evolve the input Hamiltonian through a time epsilon assuming
/// a splitting of the Hamiltonian of the form
/// \f[ \hat{H} = \frac{\hat{V}}{2} + \frac{\hat{U}}{2} + \hat{T} + \frac{\hat{U}}{2} + \frac{\hat{V}}{2}, \f]
/// where 
/// \f[ \hat{T} = \frac{ \partial H }{\partial p } \frac{ \partial }{ \partial q } \f]
/// and
/// \f[ \hat{V} = - \frac{ \partial H }{\partial q } \frac{ \partial }{ \partial p } \f]
/// are the kinetic and potential Poisson operators, respectively,
/// and \f$\hat{U}\f$ is any constraint potential operator.
/// Note that this implementation assumes that the constraint
/// potential is infinitely large so that \f$ \hat{V} \f$ vanishes
/// when the constraints are violated.
/// \param chain Input Hamiltonian chain
/// \param epsilon Time step

template <typename T>
void chainBundle<T>::evolve(T* chain, const double epsilon)
{
    
    // Initial half momentum evolution
    if(mConstraint) chain->beginEvolveP(0.5 * epsilon);
    
    // Full spatial evolution
    chain->evolveQ(epsilon);
    
    // Check constraint
    mConstraint = fConstraint(chain);
    
    // Final half momentum evolution
    if(mConstraint)  chain->finishEvolveP(0.5 * epsilon);
    else             chain->bounceP(fNormal(chain));
    
}

/// Evolve the input Hamiltonian through a time epsilon, displaying
/// the variation of the Hamiltonian after each update.
/// \see evolve
/// \param chain Input Hamiltonian chain
/// \param epsilon Time step

template <typename T>
void chainBundle<T>::verboseEvolve(T* chain, const double epsilon)
{
    
    //cout << "----------------------------------------------------" << endl;
    
    std::cout.precision(6);
    int width = 14;
    int nColumn = 4;
    
    std::cout << "Verbose Hamiltonian Evolution, Step Size = " << epsilon << ":" << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << "    "
    << std::setw(width) << std::left << "Poisson"
    << std::setw(width) << std::left << "Initial" 
    << std::setw(width) << std::left << "Current"
    << std::setw(width) << std::left << "DeltaH"
    << std::endl;
    std::cout << "    "
    << std::setw(width) << std::left << "Operator"
    << std::setw(width) << std::left << "Hamiltonian" 
    << std::setw(width) << std::left << "Hamiltonian"
    << std::setw(width) << std::left << "/ Stepsize^{2}"
    << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    
    double H0 = chain->H();
    
    // Initial half momentum evolution
    if(mConstraint) chain->beginEvolveP(0.5 * epsilon);
    
    double H1 = chain->H();
    
    std::cout << "    "
    << std::setw(width) << std::left << "hat{V}/2"
    << std::setw(width) << std::left << H0
    << std::setw(width) << std::left << H1
    << std::setw(width) << std::left << (H1 - H0) / (epsilon * epsilon)
    << std::endl;
    
    // Full spatial evolution
    chain->evolveQ(epsilon);
    
    double H2 = chain->H();
    
    std::cout << "    "
    << std::setw(width) << std::left << "hat{T}"
    << std::setw(width) << std::left << H0
    << std::setw(width) << std::left << H2
    << std::setw(width) << std::left << (H2 - H0) / (epsilon * epsilon)
    << std::endl;
    
    // Check constraint
    mConstraint = fConstraint(chain);
    
    // Final half momentum evolution
    if(mConstraint)  chain->finishEvolveP(0.5 * epsilon);
    else             chain->bounceP(fNormal(chain));
    
    double H3 = chain->H();
    
    std::cout << "    "
    << std::setw(width) << std::left << "hat{V}/2"
    << std::setw(width) << std::left << H0
    << std::setw(width) << std::left << H3
    << std::setw(width) << std::left << (H3 - H0) / (epsilon * epsilon)
    << std::endl;
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    
}

/// Initialize the dual averaging stepsize adaptation parameters,
/// and the number of leapfrog steps to ensure a proper integrationt time initialization

template <typename T>
void chainBundle<T>::initAdaptation()
{
    
    mAdaptMu = log(mStepSize / (1 - mStepSize) );
    mAdaptLogEpsBar = log(mStepSize / (1 - mStepSize) );
    
    mAdaptHBar = 0;
    mAdaptCounter = 0;
    
    mNumLeapfrog = (int)(mAdaptIntTime / mStepSize);
    mNumLeapfrog = mNumLeapfrog < 1 ? 1 : mNumLeapfrog;
    mNumLeapfrog = mNumLeapfrog > mAdaptMaxLeapfrog ? mAdaptMaxLeapfrog : mNumLeapfrog;
    
}

/// Update the dual averaging stepsize adaptation
/// \param metropolisProb Acceptance probability current proposal

template <typename T>
void chainBundle<T>::updateAdaptation(double metropolisProb)
{
    
    metropolisProb = metropolisProb > 1 ? 1 : metropolisProb;
    
    // Update averaging parameters
    ++mAdaptCounter;
    
    double delta = 1.0 / (mAdaptCounter + mAdaptT0);
    
    mAdaptHBar = (1.0 - delta) * mAdaptHBar + delta * (mAdaptTargetAccept - metropolisProb);
    
    double logEps = mAdaptMu - mAdaptHBar * sqrt(mAdaptCounter) / mAdaptGamma;
    delta = exp( - mAdaptKappa * log( mAdaptCounter ) );
    
    mAdaptLogEpsBar = (1.0 - delta) * mAdaptLogEpsBar + delta * logEps;
    
    // Then the stepsize
    mStepSize = 1.0 / ( 1.0 + exp(-mAdaptLogEpsBar) );
    
    // And the number of leapfrog steps to maintain a constant integration time
    mNumLeapfrog = (int)(mAdaptIntTime / mStepSize);
    mNumLeapfrog = mNumLeapfrog < 1 ? 1 : mNumLeapfrog;
    mNumLeapfrog = mNumLeapfrog > mAdaptMaxLeapfrog ? mAdaptMaxLeapfrog : mNumLeapfrog;
    
}

/// Transition all chains currently in the bundle

template <typename T>
void chainBundle<T>::transition()
{
    for(mIt = mChains.begin(); mIt != mChains.end(); ++mIt) transition(*mIt);
}

/// Transition the ith chain in the bundle
/// \param i Index of the chain

template <typename T>
void chainBundle<T>::transition(int i)
{
    
    if(i < 0 || i >= mChains.size())
    {
        cout << "chainBundle::transition() - Bad chain index " << i << "!" << endl;
        return;
    }
    
    transition(mChains.at(i));
    
    return;
    
}

/// Compute a trajectory for the ith chain, using the current 
/// step size and number of steps, saving the results to outputFile
/// \param outputFile File to which the output is written
/// \param i Index of the chain

template <typename T>
void chainBundle<T>::saveTrajectory(std::ofstream& outputFile, const int i)
{
    
    outputFile << mStepSize << "\t" << mNumLeapfrog << endl << endl;
    
    chain(i)->prepareEvolution();
    chain(i)->sampleP(mRandom);
    
    outputFile << 0 << "\t" << chain(i)->H() << "\t" << chain(i)->T() << "\t" << chain(i)->V() << flush;
    
    for(int j = 0; j < chain(i)->dim(); ++j) outputFile << "\t" << chain(i)->q()(j) << flush;
    for(int j = 0; j < chain(i)->dim(); ++j) outputFile << "\t" << chain(i)->p()(j) << flush;  
    outputFile << endl;
    
    for(int n = 0; n < mNumLeapfrog; ++n) 
    {
        
        evolve(chain(i), mStepSize);
        
        outputFile << n + 1 << "\t" << chain(i)->H() << "\t" << chain(i)->T() << "\t" << chain(i)->V() << flush;
        
        for(int j = 0; j < chain(i)->dim(); ++j) outputFile << "\t" << chain(i)->q()(j) << flush;
        for(int j = 0; j < chain(i)->dim(); ++j) outputFile << "\t" << chain(i)->p()(j) << flush;  
        outputFile << endl;
        
    }
    
    outputFile.close();
    
    return;
    
}

/// Generate nNumSubsample consecutive transitions from the input 
/// chain using constrained Hamiltonian Monte Carlo, storing 
/// the final state as a sample if desired.  Note that any state
/// with a NaN is immediately rejected as the probability of 
/// transitions to and from such states are defined to be zero.
///
/// \param chain Input Markov chain
/// \return Was the final proposal accepted?

template <typename T>
bool chainBundle<T>::transition(T* chain)
{
    
    // Check that the chain was properly seeded
    if(mNumSamples == 0)
    {
        
        mConstraint = fConstraint(chain);
        
        if(!mConstraint)
        {
            cout << "chainBundle::transition() - Initial chain parameters in violation "
            << "of constraints, aborting the transition!" << endl;
            return false;
        }
        
    }
    
    bool accept = true;
    
    for(int n = 0; n < mNumSubsample; ++n)
    {
        
        // Add random jitter to the step size to avoid closed loops, and
        // an occassional jerk to allow the chain to move through narrow valleys
        double stepSize = mStepSize;
        
        if(mRandom.Prob<double>(mProbStepSizeJerk)) 
        {
            stepSize *= mStepSizeJerk;
        }
        else
        {
            stepSize *= 1.0 + mStepSizeJitter * (2.0 * mRandom.FloatU() - 1.0);
        }
        
        // Prepare the chain
        chain->prepareEvolution();
        
        // Sample momenta conditioned on the current position of the chain
        chain->sampleP(mRandom);
        
        //////////////////////////////////////////////////
        //        Sample from the reject window         //
        //////////////////////////////////////////////////
        
        // Immediately accept initial point as current sample
        chain->saveAsRejectSample();
        double logSumRejectProb = -chain->H();
        
        if(mWindowSize > 0)
        {
            
            // Backwards evolution
            stepSize *= -1;
            
            chain->saveCurrentPoint();
            int s = mRandom.Integer(mWindowSize);
            
            for(int t = 0; t < s; ++t)
            {
                
                evolve(chain, stepSize);
                
                // Accept or reject updating the sample, with constraint violations immediately rejected 
                if(mConstraint) 
                {
                    double Hplus = chain->H();
                    logSumRejectProb = fLogSumExp(logSumRejectProb, -Hplus);
                    if( mRandom.Prob<double>( exp(-Hplus - logSumRejectProb) ) ) chain->saveAsRejectSample();
                }
                
            }
            
            // Forwards evolution
            stepSize *= -1;
            
            chain->restoreStoredPoint();
            
            for(int t = 0; t < mWindowSize - s - 1; ++t)
            {
                
                evolve(chain, stepSize);
                
                // Accept or reject updating the sample, with constraint violations immediately rejected 
                if(mConstraint) 
                {
                    double Hplus = chain->H();
                    logSumRejectProb = fLogSumExp(logSumRejectProb, -Hplus);
                    if( mRandom.Prob<double>( exp(-Hplus - logSumRejectProb) ) ) chain->saveAsRejectSample();
                }
                
            }
            
        }
        
        //////////////////////////////////////////////////
        //       Evolve for mNumLeapfrog iterations     //
        //////////////////////////////////////////////////
        
        if(mVerbosity >= 2)
        {
            cout << "Beginning leapfrog:" << endl;
            cout << endl;
            chain->displayState();
            cout << endl;
        }
        
        if(mTemperAlpha) chain->p() *= sqrt(mTemperAlpha);
        
        
        double m = 0.5 * (mNumLeapfrog - 2 * mWindowSize);
        
        for(unsigned int t = 0; t < mNumLeapfrog - 2 * mWindowSize; ++t)
        {
            
            // Immediately reject if a NaN shows up
            if(chain->isNaN()) 
            {
                if(mVerbosity >= 1) cout << "Immediately rejecting a NaN state..." << endl;
                break;
            }
            
            // Evolve one step
            evolve(chain, stepSize);
            
            // Temper the momenta
            if(mTemperAlpha)
            {
                if(t < m)       chain->p() *= sqrt(mTemperAlpha);
                else if(t >= m) chain->p() /= sqrt(mTemperAlpha);
            }
            
        }
        
        if(mVerbosity >= 2)
        {
            cout << "Ending leapfrog:" << endl;
            cout << endl;
            chain->displayState();
            cout << endl;
        }
        
        //////////////////////////////////////////////////
        //        Sample from the accept window         //
        //////////////////////////////////////////////////
        
        double logSumAcceptProb = 0;
        bool goodAcceptSample = false;
        
        if(!chain->isNaN())
        {
            
            // Immediately accept current point, provided the constraint is not violated            
            if(mConstraint) 
            {
                chain->saveAsAcceptSample();
                logSumAcceptProb = -chain->H();
                goodAcceptSample = true;
            }
            
            if(mWindowSize > 0)
            {
                
                for(int t = 0; t < mWindowSize - 1; ++t)
                {
                    
                    evolve(chain, stepSize);
                    
                    // Accept or reject updating the sample, with constraint violations immediately rejected 
                    if(mConstraint) 
                    {
                        
                        // Immediately reject if a NaN shows up
                        if(chain->isNaN()) 
                        {
                            if(mVerbosity >= 1) cout << "Immediately rejecting a NaN state..." << endl;
                            break;
                        }
                        
                        double Hplus = chain->H();
                        
                        // Immediately accept if no sample has yet been accepted
                        if(!goodAcceptSample)
                        {
                            logSumAcceptProb = -Hplus;
                            chain->saveAsAcceptSample();
                            goodAcceptSample = true;
                        }
                        else
                        {
                            logSumAcceptProb = fLogSumExp(logSumAcceptProb, -Hplus);
                            if( mRandom.Prob<double>( exp(-Hplus - logSumRejectProb) ) ) chain->saveAsAcceptSample();
                        }
                        
                    }
                    
                }
                
            }
            else
            {
                
                if(!mConstraint)
                {
                    chain->saveAsAcceptSample();
                    goodAcceptSample = false;
                }
                
            }
            
        }
        
        //////////////////////////////////////////////////
        //   Sample between reject and accept windows   //
        //////////////////////////////////////////////////
        
        double metropolisProb = 0;
        if(goodAcceptSample) metropolisProb = exp(logSumAcceptProb - logSumRejectProb);
        if(metropolisProb != metropolisProb) metropolisProb = 0;
        
        if(metropolisProb > 1) accept = true;
        else                   accept = mRandom.Prob<double>(metropolisProb);
        
        chain->sampleWindows(accept);
        
        if(mVerbosity >= 2)
        {
            if(accept) cout << "Accepting proposed sample with" << endl;
            else       cout << "Rejecting proposed sample with" << endl;
            
            cout << "    log(p_{accept window}) = " << logSumAcceptProb << endl;
            cout << "    log(p_{reject window}) = " << logSumRejectProb << endl;
            cout << "    p(accept window) = " << metropolisProb << endl << endl;
            
        }
        
        chain->updateMetroStats(accept, metropolisProb > 1 ? 1.0 : metropolisProb);
        
        if(mAdaptStepSize) updateAdaptation(metropolisProb);
        
    }
    
    // Store sample if desired
    if(mStoreSamples) 
    {
        mSamples.push_back(VectorXd(chain->q()));
        mSampleE.push_back(chain->V());
    }
    
    ++mNumSamples;
    
    return accept;
    
}

/// Check the accumulated error in the symplectic integrator
/// implemented in the ith chain by evolving the chain
/// n time steps, displaying the error every f steps.
/// \param i Index of the chain
/// \param N Number of time steps
/// \param f Frequncy of output

template <typename T>
void chainBundle<T>::checkIntError(int i, int N, int f)
{
    
    if(i < 0 || i >= mChains.size())
    {
        cout << "chainBundle::checkIntError() - Bad chain index " << i << "!" << endl;
        return;
    }
    
    // Prepare chain
    mChains.at(i)->prepareEvolution();
    mChains.at(i)->sampleP(mRandom);
    
    mChains.at(i)->displayState();
    
    double initH = mChains.at(i)->H();
    
    // Display Header
    std::cout.precision(6);
    int width = 12;
    int nColumn = 4;
    
    std::cout << "Displaying the integration error of chain " << i << ":" << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << "    "
    << std::setw(width) << std::left << "Step" 
    << std::setw(width) << std::left << "H_{0}"
    << std::setw(width) << std::left << "H_{i}"
    << std::setw(width) << std::left << "H_{0} - H_{i}"
    << std::endl;
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    
    // Evolve and display results
    for(int n = 0; n < N; ++n)
    {
        
        evolve(mChains.at(i), mStepSize);
        
        if(!(n % f))
        {
            std::cout << "    "
            << std::setw(width) << std::left << n
            << std::setw(width) << std::left << initH
            << std::setw(width) << std::left << mChains.at(i)->H()
            << std::setw(width) << std::left << initH -  mChains.at(i)->H()
            << std::endl;
        }
        
    }
    
    std::cout << "    " << std::setw(nColumn * width) << std::setfill('-') << "" << std::setfill(' ') << std::endl;
    std::cout << std::endl;
    
}

/// Seed all chains, using the same bounds for all components
/// \param min Minimum bound for all components
/// \param max Maximum bound for all components

template <typename T>
void chainBundle<T>::seed(const double min, const double max)
{
    for(int i = 0; i < mChains.at(0)->dim(); ++i) seed(i, min, max);
}

/// Seed the ith component of the feature space for all chains
/// \param i The selected component of the feature space
/// \param min Minimum bound for the ith component
/// \param max Maximum bound for the ith component

template <typename T>
void chainBundle<T>::seed(const int i, const double min, const double max)
{
    
    if(i < 0 || i >= mChains.at(0)->dim())
    {
        cout << "chainBundle::seed() - Bad parameter index!" << endl;
        return;
    }
    
    for(mIt = mChains.begin(); mIt != mChains.end(); ++mIt) 
    {
        (*mIt)->q()(i) = (max - min) * mRandom.FloatU<double>() + min;
    }
    
}

/// Burn in all chains until the Gelman-Rubin convergence critieria has been satisfied
///
/// Gelman, A.
/// Inference and monitoring convergence,
/// "Markov Chain Monte Carlo in Practice"
/// (1996) Chapman & Hall/CRC, New York
///
/// \param nBurn Number of burn in iterations
/// \param nCheck Number of samples for diagnosing burn in
/// \param minR Minimum Gelman-Rubin statistic
/// \param verbose Set verbose output, defaults to true

template <typename T>
void chainBundle<T>::burn(const int nBurn, const int nCheck, const double minR, const bool verbose)
{
    
    // Convergence storage
    long dim = mChains.at(0)->dim();
    double m = (double)mChains.size();
    double n = (double)nCheck;
    
    VectorXd R(dim);   // Gelman-Rubin statistics
    
    VectorXd* chainSum[(int)m];
    for(int i = 0; i < m; ++i) chainSum[i] = new VectorXd(dim);
    VectorXd ensembleSum(dim);
    VectorXd ensembleSum2(dim);
    VectorXd chainSum2(dim);
    
    bool burning = true;
    
    int nPerLine = 5;
    int nLines = (int)dim / nPerLine;
    
    cout.precision(6);
    int width = 12;
    int totalWidth = width + 8;
    int borderWidth = ((int)dim < nPerLine ? (int)dim : nPerLine) * totalWidth + 4;
    
    int acceptWidth = 14;
    
    //////////////////////////////////////////////////
    //              Burn in Markov Chain            //
    //////////////////////////////////////////////////
    
    // Temporarily store samples
    bool sampleFlag = mStoreSamples;
    mStoreSamples = false;
    
    // Burn through initial samples
    if(verbose) cout << endl << "Burn, baby, burn" << flush;
    
    for(int i = 0; i < nBurn; ++i) 
    {
        transition();
        if(verbose) if(i % 10 == 0) cout << "." << flush;
    }
    
    if(verbose) cout << endl << endl;
    if(verbose) cout << "Computing diagnostics:" << endl;
    if(verbose) cout << "    " << std::setw(borderWidth) << std::setfill('-') << "" << std::setfill(' ') << endl;
    
    while(burning)
    {
        
        ensembleSum.setZero();
        ensembleSum2.setZero();
        for(int i = 0; i < m; ++i) (chainSum[i])->setZero();
        chainSum2.setZero();
        
        // Burn through diagnostic samples
        for(int i = 0; i < nCheck; ++i) 
        {
            
            transition();
            
            mIt = mChains.begin();
            for(int j = 0; j < m; ++j, ++mIt)
            {
                
                VectorXd& q = (*mIt)->q();
                
                ensembleSum += q;
                ensembleSum2 += q.cwiseAbs2();
                
                *(chainSum[j]) += q;
                
            }
            
        }
        
        // Compute Gelman-Rubin statistic for each parameter
        double a = (n - 1) / n;
        double b = m / (m - 1);
        
        for(int j = 0; j < m; ++j)
        {
            chainSum2 += (chainSum[j])->cwiseAbs2();
        }
        
        ensembleSum = ensembleSum.cwiseAbs2();
        ensembleSum = ensembleSum.cwiseQuotient(chainSum2);
        ensembleSum /= m;
        ensembleSum = VectorXd::Ones(dim) - ensembleSum;
        
        ensembleSum2 = ensembleSum2.cwiseQuotient(chainSum2);
        ensembleSum2 *= n;
        ensembleSum2 = VectorXd::Ones(dim) - ensembleSum2;
        
        R = a * (VectorXd::Ones(dim) - b * ensembleSum.cwiseQuotient(ensembleSum2));
        
        burning = false;
        for(int i = 0; i < dim; ++i) burning |= (R[i] > minR);
        
        // Failure output
        if(verbose && burning)
        {
            cout << "    Diagnostic test failed," << endl;
            
            int k = 0;
            for(int i = 0; i < nLines; ++i)
            {
                
                cout << "        " << flush;
                for(int j = 0; j < nPerLine; ++j, ++k)
                {
                    cout << "R_{" << k << "} = " << std::setw(width) << std::left << R[k] << flush;
                }
                
                cout << endl;
                
            }
            
            if(dim % nPerLine > 0)
            {
                
                cout << "        " << flush;
                for(int i = 0; i < dim % nPerLine; ++i, ++k)
                {
                    cout << "R_{" << k << "} = " << std::setw(width) << std::left << R[k] << flush;
                }
                cout << endl;
                
            }
            
            cout << "    " << std::setw(borderWidth) << std::setfill('-') << "" << std::setfill(' ') << endl;
            
        }
        
    }
    
    // Display convergence details if desired
    if(verbose)
    {
        
        cout << "    Markov chains converged with" << endl;
        
        int k = 0;
        for(int i = 0; i < nLines; ++i)
        {
            
            cout << "        " << flush;
            for(int j = 0; j < nPerLine; ++j, ++k)
            {
                cout << "R_{" << k << "} = " << std::setw(width) << std::left << R[k] << flush;
            }
            
            cout << endl;
            
        }
        
        if(dim % nPerLine > 0)
        {
            
            cout << "        " << flush;
            for(int i = 0; i < dim % nPerLine; ++i, ++k)
            {
                cout << "R_{" << k << "} = " << std::setw(width) << std::left << R[k] << flush;
            }
            cout << endl;
            
        }
        
        cout << "    " << std::setw(borderWidth) << std::setfill('-') << "" << std::setfill(' ') << endl;
        
        cout << endl;
        
        cout << "    " << std::setw(5 * acceptWidth / 2) << std::setfill('-') << "" << std::setfill(' ') << endl;
        cout << "    "
        << std::setw(acceptWidth / 2) << std::left << "Chain" 
        << std::setw(acceptWidth) << std::left << "Global"
        << std::setw(acceptWidth) << std::left << "Local"
        << std::endl;
        cout << "    "
        << std::setw(acceptWidth / 2) << std::left << "" 
        << std::setw(acceptWidth) << std::left << "Accept Rate"
        << std::setw(acceptWidth) << std::left << "Accept Rate"
        << std::endl;
        cout << "    " << std::setw(5 * acceptWidth / 2) << std::setfill('-') << "" << std::setfill(' ') << endl;
        
        for(int i = 0; i < mChains.size(); ++i)
        {
            cout << "    "
            << std::setw(acceptWidth / 2) << std::left << i 
            << std::setw(acceptWidth) << std::left << (mChains.at(i))->acceptRate()
            << std::setw(acceptWidth) << std::left << (mChains.at(i))->movingAcceptRate()
            << endl;
        }
        cout << "    " << std::setw(5 * acceptWidth / 2) << std::setfill('-') << "" << std::setfill(' ') << endl;
        cout << endl;
        
    }
    
    mChar = true;
    
    // Return sample flag
    mStoreSamples = sampleFlag;
    
    return;
    
}

/// Compute the logarithm of the sum of two exponentials,
/// \f$ \log( \exp(a) + \exp(b) ) \f$
/// param a Argument of one exponential
/// param b Argument of the second exponential
/// return logarithm of the summed exponentials

template <typename T>
double chainBundle<T>::fLogSumExp(double a, double b)
{
    
    if(a > b) 
    {
        return a + log( 1.0 + exp(b - a) );
    }
    else
    {      
        return b + log( 1.0 + exp(a - b) );
    }
    
}

#define _BETA_CHAINBUNDLE_
#endif