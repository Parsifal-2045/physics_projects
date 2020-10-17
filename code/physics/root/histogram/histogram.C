#include <TROOT.h>
#include <TStyle.h>
#include <TH1.h>
#include <TCanvas.h>
#include <iostream>

void setStyle()
{
    gROOT->SetStyle("Plain");
    gStyle->SetPalette(57);
    gStyle->SetOptTitle(0);
}

void Histo(int ngen = 10E6)
{
    TH1F *h = new TH1F("h", "Test histogram", 100, -5., 5.);

    h->SetFillColor(kBlue);
    h->SetLineColor(kBlack);
    h->GetXaxis()->SetTitle("x");
    h->GetYaxis()->SetTitle("entries");

    h->FillRandom("gaus", ngen);

    TCanvas *c1 = new TCanvas("c1");
    
    h->Draw("H");
    h->Draw("E,P,SAME");

    double hEntries = h->GetEntries();
    double hMean = h->GetMean();
    double hRMS = h->GetRMS();
    double hMeanError = h->GetMeanError();
    double hRMSError = h->GetRMSError();

    std::cout << "Entries: " << hEntries << '\n';
    std::cout << "Mean: " << hMean << " +/- " << hMeanError << '\n';
    std::cout << "RMS: " << hRMS << " +/- " << hRMSError << '\n';
}