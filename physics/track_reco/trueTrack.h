#ifndef TRUETRACK_H
#define TRUETRACK_H
class trueTrack
{
public:
    trueTrack();
    trueTrack(double A, double B);
    void AddHit();
    double GetA() const;
    double GetB() const;
    double GetNHits() const; // number of hits on the detector
    void SetA(double A);
    void SetB(double B);

private:
    double fA, fB; // The true track parameters
    int fNHits;    // number of hits released by the track on the detector
};
#endif
