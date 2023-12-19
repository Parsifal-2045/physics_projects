#ifndef RECOTRACK_H
#define RECOTRACK_H
#include "TMatrix.h"

class recoTrack
{
public:
    recoTrack();
    ~recoTrack();
    void AddHit(int index);
    double GetA() const;
    double GetB() const;
    double GetdA() const;
    double GetdB() const;
    double GetStateA() const;
    double GetStateB() const;
    double GetStatedA() const;
    double GetStatedB() const;
    double GetchiSeed() const;
    double GetNHits() const;
    bool IsGoodSeed() const; // seed formed with 3 points from the outer layers all with the same label
    bool IsFullReco() const; // all 10 hits share the same label
    bool IsGoodReco() const; // passes the quality check (usually more than 80% of the hits share the same label)
    int *GetHits() const;
    void PrintLabels() const;
    void SetA(double A);
    void SetB(double B);
    void SetdA(double dA);
    void SetdB(double dB);
    void SetchiSeed(double chiSeed);
    void InitCovariance(TMatrix &cov);
    void InitStateVector(TMatrix &x);
    void InitStatex(double xState);
    void PropagateAndFilter(double xProp, double yMeas = 0, bool print = false);
    void PrintCovariance() const;
    void PrintStateVector() const;
    void PrintStatex() const;
    void SetGoodSeed();
    void SetFullReco();
    void SetGoodReco();

private:
    double fA, fB, fdA, fdB; // the reco track parameter
    double fchiSeed;         // the Chi**2/NDF of the seed
    TMatrix *fCov;           // the state covariance
    TMatrix *fX;             // The state vector
    double fxCoord;          // the x coordinate of the state
    int *fHits;              // pointer to the array of hits gen labels (max size 10 as the detector planes)
    int fNHits;              // the number of hits in the track
    Bool_t fIsGoodSeed;
    Bool_t fIsFullReco;
    Bool_t fIsGoodReco;
};
#endif
