#include <iostream>

#include "recoTrack.h"
#include "TMath.h"

recoTrack::recoTrack()
{
    fNHits = 0;
    fHits = new int[10];
    fIsGoodSeed = false;
    fIsGoodReco = false;
    fIsFullReco = false;
    fCov = new TMatrix(2, 2);
    fX = new TMatrix(2, 1);
}

recoTrack::~recoTrack()
{
    delete[] fHits;
    delete fCov;
    delete fX;
}

void recoTrack::AddHit(int index)
{
    if (fNHits < 10)
    {
        fHits[fNHits] = index;
        fNHits++;
    }
    else
    {
        std::cout << " Could not add the hit" << std::endl;
    }
}
void recoTrack::PrintLabels() const
{
    for (Int_t i = 0; i < fNHits; i++)
        std::cout << "Track label = [ " << i << "] is:" << fHits[i] << std::endl;
}
double recoTrack::GetA() const { return fA; }
double recoTrack::GetB() const { return fB; }
double recoTrack::GetdA() const { return fdA; }
double recoTrack::GetdB() const { return fdB; }
double recoTrack::GetStateA() const { return (*fX)(0, 0); }
double recoTrack::GetStateB() const { return (*fX)(1, 0); }
double recoTrack::GetStatedA() const { return TMath::Sqrt(TMath::Abs((*fCov)(0, 0))); }
double recoTrack::GetStatedB() const { return TMath::Sqrt(TMath::Abs((*fCov)(1, 1))); }
double recoTrack::GetchiSeed() const { return fchiSeed; }

double recoTrack::GetNHits() const { return fNHits; }
bool recoTrack::IsGoodSeed() const { return fIsGoodSeed; }
bool recoTrack::IsGoodReco() const { return fIsGoodReco; }
bool recoTrack::IsFullReco() const { return fIsFullReco; }
int *recoTrack::GetHits() const { return fHits; }
void recoTrack::SetA(double A) { fA = A; }
void recoTrack::SetB(double B) { fB = B; }
void recoTrack::SetdA(double dA) { fdA = dA; }
void recoTrack::SetdB(double dB) { fdB = dB; }
void recoTrack::SetchiSeed(double chiSeed) { fchiSeed = chiSeed; }
void recoTrack::SetGoodSeed() { fIsGoodSeed = true; }
void recoTrack::SetFullReco() { fIsFullReco = true; }
void recoTrack::SetGoodReco() { fIsGoodReco = true; }
void recoTrack::InitCovariance(TMatrix &cov) { *fCov = cov; }
void recoTrack::InitStateVector(TMatrix &x) { *fX = x; }
void recoTrack::InitStatex(double xState) { fxCoord = xState; }
void recoTrack::PrintCovariance() const { fCov->Print(); }
void recoTrack::PrintStateVector() const { fX->Print(); }
void recoTrack::PrintStatex() const { std::cout << "State was propagated at x = " << fxCoord << std::endl; }

void recoTrack::PropagateAndFilter(double xProp, double yMeas, bool print)
{

    // Kalman Filter (Prediction step)
    TMatrix F(2, 2);
    F(0, 0) = 1;
    F(0, 1) = xProp - fxCoord;
    F(1, 0) = 0;
    F(1, 1) = 1; // for System equation, F matrix
    TMatrix H(1, 2);
    H(0, 0) = 1;
    H(0, 1) = 0; // for Measurement equation, H matrix
    TMatrix FT(2, 2);
    FT.Transpose(F);
    TMatrix HT(2, 1);
    HT.Transpose(H);

    TMatrix xk(2, 1);
    TMatrix xk_prop(2, 1);
    TMatrix xk_filter(2, 1); // state vector, redicted state and filtered state
    TMatrix Ck(2, 2);
    TMatrix Ck_prop(2, 2);
    TMatrix Ck_prop_inv(2, 2); // correspondig covariance
    TMatrix Ck_filter(2, 2);
    TMatrix Ck_filter_inv(2, 2);

    double invV = 0;
    if (TMath::Abs(yMeas) > 0)
    {
        invV = 1. / 0.2 / 0.2; // if a measurement is available define measurement covariance matrix: 1 / sigma^2
    }
    double M = yMeas; // the measurement

    TMatrix HTinvVm(2, 1);
    HTinvVm(0, 0) = M * invV;
    HTinvVm(1, 0) = 0;
    TMatrix HTinvVH(2, 2);
    HTinvVH(0, 0) = invV;
    HTinvVH(0, 1) = 0;
    HTinvVH(1, 0) = 0;
    HTinvVH(1, 1) = 0;

    xk = *fX;   // current state
    Ck = *fCov; // input from the internal covariance
    if (print)
    {
        std::cout << "fxCoord =" << fxCoord << std::endl;
        std::cout << "Measurement = " << yMeas << std::endl;
        std::cout << "Initial state vector and covariance " << std::endl;
        xk.Print();
        Ck.Print();
    }

    // propagation
    xk_prop = F * xk;      // predicted state
    Ck_prop = F * Ck * FT; // predicted covariance

    if (print)
    {
        std::cout << "Predicted state vector and covariance " << std::endl;
        xk_prop.Print();
        Ck_prop.Print();
    }

    // filter
    Ck_prop_inv = Ck_prop;
    Ck_prop_inv.Invert();

    // filtered state
    xk_filter = (Ck_prop_inv * xk_prop + HTinvVm); // numerator
    Ck_filter_inv = (Ck_prop_inv + HTinvVH);       // denominator
    Ck_filter = Ck_filter_inv.Invert();
    xk_filter = Ck_filter * xk_filter;

    if (print)
    {
        std::cout << "Filtered state vector and covariance " << std::endl;
        xk_filter.Print();
        Ck_filter.Print();
    }

    // update the internal state
    *fX = xk_filter;
    // update internal Covariance
    *fCov = Ck_filter;
    // update x propagation coordinate of the state
    fxCoord = xProp;

    if (print)
        std::cout << " Final fxCoord =" << fxCoord << std::endl;

    return;
}
