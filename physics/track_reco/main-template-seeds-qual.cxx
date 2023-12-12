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

  // booking some histos

  // for acceptance
  TH1F *hNHits = new TH1F("hNHits", "Number of hits, generated track ", 20, -0.5, 19.5);
  TH2F *hHits = new TH2F("hHits", " Hits map ", 1500, 0, 150, 2000, -100, 100);
  TH1F *hBDen = new TH1F("hBDen", " B parameter, all gen tracks", 100, -1., 1.);
  TH1F *hBNum = new TH1F("hBNum", " B parameter, tracks having at least 8 hits", 100, -1., 1.);
  hBNum->Sumw2();
  hBDen->Sumw2();

  // graphs/function for seeding part
  TGraphErrors *graph = new TGraphErrors();         // auxilary TGraph used in seeding
  TF1 *line = new TF1("line", "[0]+[1]*x", 0, 150); // linear function in seeding

  TGraph *graphChi = new TGraph(); // performance plot on X**2 cut
  graphChi->SetTitle("optimal cut in seeding; seeding efficiency; seeding purity");

  TH1F *hFit = new TH1F("hFit", " Seed Fit quality", 1000, 0, 100);
  TH1F *hFitGood = new TH1F("hFitGood", " Good Seed Fit quality", 1000, 0, 100);

  const Int_t nEvt = 100; // generate nEvt events

  // Detector dimensions
  // x range goes from 0 to 1.5 m. Planes (10) start at 20 cm, 10 cms apart [20 cm,..,110 cm]
  // They are 1m along y. Full detector acceptance for straight tracks coming from the vertex
  // is about +/- 25 degrees in angle

  const Int_t nPlanes = 10;
  const Double_t HalfPlaneWidth = 50; // cm
  Double_t xPlanes[nPlanes];          // planes x position
  for (Int_t i = 0; i < nPlanes; i++)
    xPlanes[i] = 20. + i * 10; // in cm

  Double_t A = 0, B = 0;
  Double_t xImpact, yImpact;

  // for seeding
  // Double_t maxChi=100.;
  double maxChi[10] = {0.5, 1., 2., 3., 4., 5., 6., 7., 8., 10}; // reference X**2 values

  for (Int_t iChi = 0; iChi < 10; iChi++)
  {

    int nTotSeed = 0;
    int nTotGoodSeed = 0;

    // start loop over events

    for (Int_t ievt = 0; ievt < nEvt; ievt++)
    {

      Int_t nHits = 0;                                        // total hits per event
      Int_t nHitsP[nPlanes] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // number of hits per plane
      trueTrack genTrack[100];                                // max 100 tracks per event
      hit Hits[nPlanes][100];                                 // Matrix of Hits
      for (Int_t i = 0; i < nPlanes; i++)
        nHitsP[i] = 0;

      //_____________________________________________
      //
      // The true tracks and detector hits generation
      //_____________________________________________
      //
      for (Int_t iTrk = 0; iTrk < nTrk; iTrk++)
      {

        // generate the track
        A = gRandom->Uniform(-2, 2);
        B = gRandom->Uniform(-0.3, 0.3); // about 17 deg
        //   B=gRandom->Uniform(-50./110,50./110); //about 25 deg
        //  B=gRandom->Uniform(-1.,1.);//45 deg

        genTrack[iTrk].SetA(A);
        genTrack[iTrk].SetB(B);

        // now loop over the detector planes and add the hits
        //_____________________________________________
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
          hBNum->Fill(B);

        hBDen->Fill(B);

      } // close loop on generated tracks

      // Seeding Part, now

      Int_t nSeed = 0;
      recoTrack Track[1000]; // allow for a larger size (fake seeds)

      // add a first constraint: the vertex
      graph->SetPoint(0, 0, 0);
      graph->SetPointError(0, 0.1, 2);
      // now loop on the three last external planes (seeding from outside)
      for (Int_t i1 = 0; i1 < nHitsP[nPlanes - 1]; i1++)
      {
        for (Int_t i2 = 0; i2 < nHitsP[nPlanes - 2]; i2++)
        {
          for (Int_t i3 = 0; i3 < nHitsP[nPlanes - 3]; i3++)
          {
            graph->SetPoint(1, Hits[nPlanes - 1][i1].GetX(), Hits[nPlanes - 1][i1].GetY());
            graph->SetPoint(2, Hits[nPlanes - 2][i2].GetX(), Hits[nPlanes - 2][i2].GetY());
            graph->SetPoint(3, Hits[nPlanes - 3][i3].GetX(), Hits[nPlanes - 3][i3].GetY());
            graph->SetPointError(1, 0.1, 0.2);
            graph->SetPointError(2, 0.1, 0.2);
            graph->SetPointError(3, 0.1, 0.2);
            bool status = graph->Fit(line, "Q");
            hFit->Fill(line->GetChisquare() / line->GetNDF()); // Fit quality
            graph->Clear();
            if (!status && (line->GetChisquare() / line->GetNDF() < maxChi[iChi]))
            { // Take this as a good seed (X**2/NDF<maxChi)
              //  if(!status&& (line->GetChisquare()/line->GetNDF()<maxChi)){ //Take this as a good seed (X**2/NDF<maxChi)
              Track[nSeed].SetA(line->GetParameter(0));
              Track[nSeed].SetB(line->GetParameter(1));
              Track[nSeed].SetdA(line->GetParError(0));
              Track[nSeed].SetdB(line->GetParError(1));
              Track[nSeed].SetchiSeed(line->GetChisquare() / line->GetNDF());
              Track[nSeed].AddHit(Hits[nPlanes - 1][i1].GetTrackIndex());
              Track[nSeed].AddHit(Hits[nPlanes - 2][i2].GetTrackIndex());
              Track[nSeed].AddHit(Hits[nPlanes - 3][i3].GetTrackIndex());
              nSeed++;
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
              badSeed = true;
          }
          if (!badSeed)
          {
            nGoodSeed++;                              // if all track labels are that of the track, good seed found
            hFitGood->Fill(Track[iTrk].GetchiSeed()); // Fit quality good seeds
            break;
          }
        }
      }

      nTotSeed += nSeed;
      nTotGoodSeed += nGoodSeed;

    } // close loop over the events

    std::cout << "number of total seeds = " << nTotSeed << std::endl;
    std::cout << "number of total good seeds = " << nTotGoodSeed << std::endl;
    std::cout << "Seeding efficiency = " << float(nTotGoodSeed) / nTrk / nEvt << std::endl;
    std::cout << "Seeding purity = " << float(nTotGoodSeed) / nTotSeed << std::endl;

    graphChi->SetPoint(iChi, float(nTotGoodSeed) / nTrk / nEvt, float(nTotGoodSeed) / nTotSeed);

  } // close loop on chiSquare ref values

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

  TCanvas *cPlot2 = new TCanvas();
  hFit->Draw();
  hFitGood->SetFillColor(kBlue);
  hFitGood->Draw("same");

  TCanvas *cPlot3 = new TCanvas();
  graphChi->SetMarkerStyle(20);
  graphChi->SetMarkerColor(kRed);
  graphChi->SetMarkerSize(1.);
  graphChi->Draw("APE");

  return;
}
