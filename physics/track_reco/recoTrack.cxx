#include <iostream>

#include "TMath.h"
#include "recoTrack.h"

recoTrack::~recoTrack() { delete[] fHits; }
recoTrack::recoTrack()
{
    fNHits = 0;
    fHits = new int[10];
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
        std::cout << " could not add the hit" << std::endl;
    }
}
void recoTrack::PrintLabels() const
{
    for (Int_t i = 0; i < fNHits; i++)
        std::cout << "track label = [ " << i << "] is:" << fHits[i] << std::endl;
}
double recoTrack::GetA() const { return fA; }
double recoTrack::GetB() const { return fB; }
double recoTrack::GetdA() const { return fdA; }
double recoTrack::GetdB() const { return fdB; }
double recoTrack::GetchiSeed() const { return fchiSeed; }
double recoTrack::GetNHits() const { return fNHits; }
int *recoTrack::GetHits() const { return fHits; }
void recoTrack::SetA(double A) { fA = A; }
void recoTrack::SetB(double B) { fB = B; }
void recoTrack::SetdA(double dA) { fdA = dA; }
void recoTrack::SetdB(double dB) { fdB = dB; }
void recoTrack::SetchiSeed(double chiSeed) { fchiSeed = chiSeed; }
