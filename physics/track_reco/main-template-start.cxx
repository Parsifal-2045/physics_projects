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
  TH1F *hBNum = new TH1F("hBNum", " B parameter, tracks having 8 hits", 100, -1., 1.); // 80% acceptance
  hBNum->Sumw2();
  hBDen->Sumw2();

  // graphs/function for seeding part
  TGraphErrors *graph = new TGraphErrors();         // used in seeding used to fit the candidate tracks and get their chi2
  TF1 *line = new TF1("line", "[0]+[1]*x", 0, 150); // linear function for track formation in seeding step

  TH1F *hFit = new TH1F("hFit", " Seed Fit quality", 1000, 0, 100);              // chi2 of all seeds
  TH1F *hFitGood = new TH1F("hFitGood", " Good Seed Fit quality", 1000, 0, 100); // chi2 of seeds that come from a single generated track (check that all labels of the track's points are the same)

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
    xPlanes[i] = 20. + i * 10; // x position in cm (convert detector layer to distance)
  }

  Double_t A = 0, B = 0;
  Double_t xImpact, yImpact;

  // seeding parameters
  Double_t maxChi = 100.;
  int nTotSeed = 0;
  int nTotGoodSeed = 0;

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
      // generate the track
      A = gRandom->Uniform(-2, 2);
      B = gRandom->Uniform(-0.3, 0.3); // about 17 degrees
      // B = gRandom->Uniform(-50. / 110., 50. / 110.); // exact detector's acceptance (about 25 degrees)
      // B = gRandom->Uniform(-1., 1.); // 45 degrees

      genTrack[iTrk].SetA(A);
      genTrack[iTrk].SetB(B);

      // loop on the planes  to count how many hits in the det
      for (Int_t i = 0; i < nPlanes; i++)
      {
        xImpact = xPlanes[i];
        yImpact = A + B * xImpact;
        if (TMath::Abs(yImpact) < HalfPlaneWidth)
        {
          genTrack[iTrk].AddHit();                                           // increment # hits in genTrack
          Hits[i][nHitsP[i]].Set(xImpact, yImpact, iTrk);                    // set x,y,gen track label in current hit
          hHits->Fill(Hits[i][nHitsP[i]].GetX(), Hits[i][nHitsP[i]].GetY()); // hit map
          nHits++;
          nHitsP[i]++;
        }
      } // close loop on detector layers

      hNHits->Fill(genTrack[iTrk].GetNHits()); // fill # hits for current track

      // monitor fraction of reference tracks as a function of B (acceptance)
      if (genTrack[iTrk].GetNHits() >= 8)
      {
        hBNum->Fill(B);
      }
      hBDen->Fill(B);
    } // close loop on generated tracks

    //_____________________________________________
    // Seeding
    //_____________________________________________

    Int_t nSeed = 0;
    recoTrack Track[1000]; // allow for a larger size (fake seeds)

    // add a first constraint: the vertex
    graph->SetPoint(0, 0, 0);
    graph->SetPointError(0, 0.1, 2); // nominal vertex, 2cm beam-spot size
    // loop on the three last external planes (seeding from outside)
    for (Int_t i1 = 0; i1 < nHitsP[nPlanes - 1]; i1++)
    {
      for (Int_t i2 = 0; i2 < nHitsP[nPlanes - 2]; i2++)
      {
        for (Int_t i3 = 0; i3 < nHitsP[nPlanes - 3]; i3++)
        {
          graph->SetPoint(1, Hits[nPlanes - 1][i1].GetX(), Hits[nPlanes - 1][i1].GetY());
          graph->SetPoint(2, Hits[nPlanes - 2][i2].GetX(), Hits[nPlanes - 2][i2].GetY());
          graph->SetPoint(3, Hits[nPlanes - 3][i3].GetX(), Hits[nPlanes - 3][i3].GetY());
          graph->SetPointError(1, 0.1, 0.2); // uncertainty on detector planes "no width" and 2mm along y
          graph->SetPointError(2, 0.1, 0.2);
          graph->SetPointError(3, 0.1, 0.2);
          bool status = graph->Fit(line, "Q");               // status is 0 if the fit converges
          hFit->Fill(line->GetChisquare() / line->GetNDF()); // fit quality for all seeds
          graph->Clear();
          if (!status && (line->GetChisquare() / line->GetNDF() < maxChi)) // fit converged and the chi2 is less than the limit set before
          {                                                                // take this as a good seed (X**2/NDF<maxChi)
            Track[nSeed].SetA(line->GetParameter(0));                      // set parameters of the reconstucted track
            Track[nSeed].SetB(line->GetParameter(1));
            Track[nSeed].SetdA(line->GetParError(0));
            Track[nSeed].SetdB(line->GetParError(1));
            Track[nSeed].SetchiSeed(line->GetChisquare() / line->GetNDF()); // set chi2 of the reconstructed track
            Track[nSeed].AddHit(Hits[nPlanes - 1][i1].GetTrackIndex());     // add hits to the reconstructed track
            Track[nSeed].AddHit(Hits[nPlanes - 2][i2].GetTrackIndex());
            Track[nSeed].AddHit(Hits[nPlanes - 3][i3].GetTrackIndex());
            nSeed++; // move to the next seed
          }
        }
      }
    }

    // Now check how many seeds have the right hit assignment
    bool badSeed = false;
    int nGoodSeed = 0;
    int *labels = new int[nPlanes];
    for (Int_t igenTrk = 0; igenTrk < nTrk; igenTrk++)
    { // loop on gen tracks
      for (Int_t iTrk = 0; iTrk < nSeed; iTrk++)
      { // loop on seeds
        badSeed = false;
        labels = Track[iTrk].GetHits();
        for (Int_t i = 0; i < Track[iTrk].GetNHits(); i++)
        {
          if (igenTrk != labels[i])
          {
            badSeed = true; // 1 wrong label makes the seed bad
          }
        }
        if (!badSeed)
        {
          nGoodSeed++;                              // if all track labels are that of the track, good seed found
          hFitGood->Fill(Track[iTrk].GetchiSeed()); // fit quality for good seeds
          break;
        }
      }
    }

    nTotSeed += nSeed;
    nTotGoodSeed += nGoodSeed;

  } // close loop over the events

  // monitor acceptance
  TH1F *hAcc = new TH1F(*hBDen);
  hAcc->SetTitle("Acceptance vs B, at least 8 hits");
  hAcc->Divide(hBNum, hBDen, 1, 1, "B"); // acceptance = # tracks with at least 8 hits / total # of generated tracks

  TCanvas *cPlot1 = new TCanvas();
  cPlot1->Divide(2, 2);
  cPlot1->cd(1);
  hNHits->Draw();
  cPlot1->cd(2);
  hHits->Draw();
  cPlot1->cd(3);
  hAcc->Draw();

  // monitor seeding
  std::cout << "Total number of seeds = " << nTotSeed << std::endl;
  std::cout << "Total number of good seeds = " << nTotGoodSeed << std::endl;
  std::cout << "Seeding efficiency = " << float(nTotGoodSeed) / nTrk / nEvt << std::endl; // number of good seeds / (number of tracks per event * number of events)
  std::cout << "Seeding purity = " << float(nTotGoodSeed) / nTotSeed << std::endl;        // number of good seeds / total number of seeds
  // 100 is way too large as a cut on chi2 -> leads to low purity
  // check cuts around chi2 = 10 with seeds-qual and settle on a better cut
  // selected chi2 cut at 4 in the end

  TCanvas *cPlot2 = new TCanvas();
  hFit->Draw();
  hFitGood->SetFillColor(kBlue);
  hFitGood->Draw("same");
}
