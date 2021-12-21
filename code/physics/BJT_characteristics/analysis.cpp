#include <iostream>
#include <TROOT.h>
#include <TStyle.h>
#include <TF1.h>
#include <TH1.h>
#include <TCanvas.h>
#include <TGraphErrors.h>
#include <TMath.h>
#include <TPaveStats.h>
#include <TMultiGraph.h>
#include <TFile.h>
#include <TPaveStats.h>
#include <TLegend.h>

void analysis()
{
    gROOT->SetStyle("Plain");
    gStyle->SetPalette(57);
    gStyle->SetOptFit(1111);
    TCanvas *c = new TCanvas("c", "Caratteristica BJT");
    c->SetGrid();
    auto mg = new TMultiGraph();

    TGraphErrors *min = new TGraphErrors("100.dat", "%lg %lg %lg %lg");
    min->SetTitle("Caratteristica a 100 #muA");
    min->SetLineColor(kBlue);
    min->SetMarkerColor(kBlue);
    TF1 *fitfunc = new TF1("fitfunc", "[0] + [1]*x", 1, 4);
    fitfunc->SetParameter(0, 15);
    fitfunc->SetParameter(1, 1);
    fitfunc->SetLineColor(kBlack);
    fitfunc->SetLineWidth(2);
    min->Fit("fitfunc", "R");

    TGraphErrors *mag = new TGraphErrors("200.dat", "%lg %lg %lg %lg");
    mag->SetTitle("Caratteristica a 200 #mu A");
    mag->SetLineColor(kRed);
    mag->SetMarkerColor(kRed);
    TF1 *fitfunc2 = new TF1("fitfunc2", "[0] + [1]*x", 1, 4);
    fitfunc2->SetParameter(0, 30);
    fitfunc2->SetParameter(1, 1);
    fitfunc2->SetLineColor(kBlack);
    fitfunc2->SetLineWidth(2);
    mag->Fit("fitfunc2", "R");

    mg->Add(min);
    mg->Add(mag);
    mg->GetXaxis()->SetTitle("Tensione tra collettore ed emettitore (|Vce|), [V]");
    mg->GetYaxis()->SetTitle("Corrente di collettore (|Ic|), [mA]");
    mg->Draw("APC");

    gPad->Update();
    TPaveStats *st1 = (TPaveStats *)min->FindObject("stats");
    st1->SetX1NDC(0.601719);
    st1->SetY1NDC(0.246835);
    st1->SetX2NDC(0.962751);
    st1->SetY2NDC(0.447257);

    TPaveStats *st2 = (TPaveStats *)mag->FindObject("stats");
    st2->SetX1NDC(0.601719);
    st2->SetY1NDC(0.580169);
    st2->SetX2NDC(0.962751);
    st2->SetY2NDC(0.780591);

    mg->Draw("APC,SAME");

    TLegend *leg = new TLegend(0.0988539, 0.803797, 0.442693, 0.951477, "Risposta BJT in funzione della corrente di base");
    leg->SetFillColor(0);
    leg->AddEntry(min, "Corrente di base |Ib| = 100 #muA");
    leg->AddEntry(mag, "Corrente di base |Ib| = 200 #muA");
    leg->SetTextSize(0.022);
    leg->Draw("SAME");

    c->Print("Caratteristica BJT.pdf");
}