#include <iostream>

#include "trueTrack.h"
#include "recoTrack.h"
#include "hit.h"
#include "TMath.h"
#include "TRandom.h"
#include "TH1.h"
#include "TH2.h"
#include "TF1.h"
#include "TGraphErrors.h"
#include "TCanvas.h"

void generate(int nTrk = 10)
{
  if (!(nTrk < 100))
  {
    std::cout << "too many tracks requested, exit " << std::endl;
    return;
  }

  // booking some histos for acceptance
  TH1F *hNHits = new TH1F("hNHits", "Number of hits, generated track ", 20, -0.5, 19.5);
  TH2F *hHits = new TH2F("hHits", " Hits map ", 1500, 0, 150, 2000, -100, 100);
  TH1F *hBDen = new TH1F("hBDen", " B parameter, all gen tracks", 100, -1., 1.);
  TH1F *hBNum = new TH1F("hBNum", " B parameter, tracks having 8 hits", 100, -1., 1.);
  hBNum->Sumw2();
  hBDen->Sumw2();

  const Int_t nEvt = 100; // generate nEvt events

  // Detector dimensions
  // x range goes from 0 to 1.5 m. Planes (10) start at 20 cm, 10 cms apart [20 cm,..,110 cm]
  // They are 1m along y. Full detector acceptance for straight tracks coming from the vertex
  // is about +/- 25 degrees in angle

  const Int_t nPlanes = 10;
  const Double_t HalfPlaneWidth = 50; // cm
  Double_t xPlanes[nPlanes];          // planes x position
  for (Int_t i = 0; i < nPlanes; i++)
  {
    xPlanes[i] = 20. + i * 10; // in cm
  }

  Double_t A = 0, B = 0;
  Double_t xImpact, yImpact;

  // start loop over events

  for (Int_t ievt = 0; ievt < nEvt; ievt++)
  {
    Int_t nHits = 0;                                        // total hits per event
    Int_t nHitsP[nPlanes] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // number of hits per plane
    trueTrack genTrack[100];                                // max 100 tracks per event
    hit Hits[nPlanes][100];                                 // Matrix of Hits:
    for (Int_t i = 0; i < nPlanes; i++)
    {
      nHitsP[i] = 0;
    }

    //_____________________________________________
    // The true track and detector hits generation
    //_____________________________________________

    for (Int_t iTrk = 0; iTrk < nTrk; iTrk++)
    {

      //_____________________________________________
      //
      // Add code Here
      //_____________________________________________
      //

      // now loop on the planes  to count how many hits in the det

      for (Int_t i = 0; i < nPlanes; i++)
      {

        //_____________________________________________
        //
        // Add code Here
        //_____________________________________________
        //

        //
      } // close loop on detector layers

    } // close loop on generated tracks
      //_________________________________________________
  }   // end loop over the events

  TH1F *hAcc = new TH1F(*hBDen);
  hAcc->SetTitle("Acceptance vs B, at least 8 hits");
  hAcc->Divide(hBNum, hBDen, 1, 1, "B");

  TCanvas *cPlot1 = new TCanvas();
  cPlot1->Divide(2, 2);
  cPlot1->cd(1);
  hNHits->Draw();
  cPlot1->cd(2);
  hHits->Draw();
  cPlot1->cd(3);
  hAcc->Draw();

  return;
}
