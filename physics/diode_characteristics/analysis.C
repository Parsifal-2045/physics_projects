#include <iostream>
#include <TROOT.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TGraphErrors.h>
#include <TMath.h>
#include <TPaveStats.h>

void cosm()
{
    gROOT->SetStyle("Plain");
    gStyle->SetPalette(57);
    gStyle->SetOptFit(1111);
}

void cal()
{
    TCanvas *c = new TCanvas("c", "Retta di calibrazione");
    c->SetGrid();
    TGraphErrors *cal = new TGraphErrors("calibrazione.dat", "%lg %lg %lg %lg");
    cal->SetTitle("Retta di calibrazione");
    cal->GetXaxis()->SetTitle("Misura oscilloscopio (mV)");
    cal->GetYaxis()->SetTitle("Misura multimetro (mV)");
    cal->Fit("pol1");
    TF1 *fitfunc = cal->GetFunction("pol1");
    fitfunc->SetLineColor(kRed);
    fitfunc->SetLineWidth(2);
    cal->Draw("AP");
    //c->Print("Retta di calibrazione.pdf");
}

void Si()
{
    TCanvas *c = new TCanvas("c", "Caratteristica del diodo al silicio");
    c->SetLogy();
    c->SetGrid();
    TGraphErrors *si = new TGraphErrors("Si.dat", "%lg %lg %lg %lg");
    si->SetTitle("Caratteristica del diodo al silicio");
    si->GetXaxis()->SetTitle("Voltage (mV)");
    si->GetYaxis()->SetTitle("Current (mA)");

    TF1 *fitfunc = new TF1("fitfunc", "[0]*(exp(x/[1])-1.)", 400, 800);
    fitfunc->SetParameter(0, 2e-6);
    fitfunc->SetParameter(1, 50.);
    fitfunc->SetParName(0, "I0");
    fitfunc->SetParName(1, ("#eta*VT"));
    fitfunc->SetLineColor(kRed);
    fitfunc->SetLineWidth(2);
    si->Fit("fitfunc", "R");
    si->Draw("AP");
    //c->Print("Caratteristica diodo al silicio.pdf");
}

void Ge()
{
    TCanvas *c = new TCanvas("c", "Caratteristica del diodo al germanio");
    c->SetLogy();
    c->SetGrid();
    TGraphErrors *Ge = new TGraphErrors("Ge.dat", "%lg %lg %lg %lg");
    Ge->SetTitle("Caratteristica del diodo al germanio");
    Ge->GetXaxis()->SetTitle("Voltage (mV)");
    Ge->GetYaxis()->SetTitle("Current (mA)");

    TF1 *fitfunc = new TF1("fitfunc", "[0]*(exp(x/[1])-1.)", 100, 210);
    fitfunc->SetParameter(0, 2e-6);
    fitfunc->SetParameter(1, 50.);
    fitfunc->SetParName(0, "I0");
    fitfunc->SetParName(1, ("#eta*VT"));
    fitfunc->SetLineColor(kRed);
    fitfunc->SetLineWidth(2);
    Ge->Fit("fitfunc", "R");
    Ge->Draw("AP");
    //c->Print("Caratteristica diodo al germanio.pdf");
}