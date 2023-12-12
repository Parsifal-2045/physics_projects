#ifndef RECOTRACK_H
#define RECOTRACK_H

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
    double GetchiSeed() const;
    double GetNHits() const;
    int *GetHits() const;
    void PrintLabels() const;
    void SetA(double A);
    void SetB(double B);
    void SetdA(double dA);
    void SetdB(double dB);
    void SetchiSeed(double chiSeed);

private:
    double fA, fB, fdA, fdB; // the reco track parameters: A -> intercept, B -> slope
    double fchiSeed;         // the Chi**2/NDF of the seed
    int *fHits;              // pointer to the array of hits associated to the track, stores the labels of the true track that generated the hit (max size 10 as the detector planes)
    int fNHits;              // the number of hits associated to the track. Expected 10 if all hits are in the acceptance region of the detector, could be < 10. Can be used to define acceptance
};
#endif
