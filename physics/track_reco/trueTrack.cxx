#include "trueTrack.h"

trueTrack::trueTrack()
{
    fA = 0;
    fB = 0;
    fNHits = 0;
}
trueTrack::trueTrack(double A, double B)
{
    fA = A;
    fB = B;
    fNHits = 0;
}
void trueTrack::AddHit() { fNHits++; }
double trueTrack::GetA() const { return fA; }
double trueTrack::GetB() const { return fB; }
void trueTrack::SetA(double A) { fA = A; }
void trueTrack::SetB(double B) { fB = B; }
double trueTrack::GetNHits() const { return fNHits; }
