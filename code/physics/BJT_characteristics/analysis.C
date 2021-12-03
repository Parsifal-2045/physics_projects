#include <iostream>
#include <TROOT.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TGraphErrors.h>
#include <TMath.h>
#include <TPaveStats.h>
#include<TMultiGraph.h>

void cosm()
{
    gROOT->SetStyle("Plain");
    gStyle->SetPalette(57);
    gStyle->SetOptFit(1111);
}

void analysis()
{
    TCanvas *c = new TCanvas("c", "Caratteristica BJT");
    c->SetGrid();
    auto mg = new TMultiGraph();

    TGraphErrors *min = new TGraphErrors("100.dat", "%lg %lg %lg %lg");
    min->SetTitle("Caratteristica a 100 #muA");
    TF1 *fitfunc = new TF1("fitfunc", "[0] + [1]*x", 1, 4);
    fitfunc->SetParameter(0, 15);
    fitfunc->SetParameter(1, 1);
    min->Fit("fitfunc", "R");

    TGraphErrors *mag = new TGraphErrors("200.dat", "%lg %lg %lg %lg");
    mag->SetTitle("Caratteristica a 200 #mu A");
    TF1 *fitfunc2 = new TF1("fitfunc2", "[0] + [1]*x", 1, 4);
    fitfunc2->SetParameter(0, 30);
    fitfunc2->SetParameter(1, 1);
    mag->Fit("fitfunc2", "R");

    mg->Add(min);
    mg->Add(mag);
    mg->GetXaxis()->SetTitle("Voltage (V)");
    mg->GetYaxis()->SetTitle("Current (mA)");
    mg->Draw("APC"); 
}