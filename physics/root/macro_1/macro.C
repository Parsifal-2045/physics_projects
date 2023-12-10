#include <TROOT.h>
#include <TStyle.h>
#include <TH1.h>
#include <TCanvas.h>
#include <TF1.h>
#include <iostream>

void SetStyle()
{
    gROOT->SetStyle("Plain");
    gStyle->SetPalette(57);
    gStyle->SetOptTitle(0);
}

double Myfunction(double *x, double *par)
{
    double xx = x[0];
    double val = par[0] * TMath::Exp(-(xx - par[1]) * (xx - par[1]) / 2. / par[2] / par[2]);
    return val;
}

void histoFit(int ngen = 1E8)
{
    gStyle->SetOptStat(2210);
    gStyle->SetOptFit(1111);

    char *histName = new char[10];
    TH1F *h[2];
    for (int i = 0; i != 2; i++)
    {
        sprintf(histName, "h%d", i);
        h[i] = new TH1F(histName, "Test histogram", 100, -5., 5.);

        h[i]->SetLineColor(kBlack);
        h[i]->GetXaxis()->SetTitle("x");
        h[i]->GetYaxis()->SetTitle("Entries");
        h[i]->GetYaxis()->SetTitleOffset(1.3);
        h[i]->SetMarkerStyle(kCircle);
        h[i]->SetMarkerSize(0.5);
    }

    h[0]->SetFillColor(kBlue);
    h[1]->SetFillColor(kRed);

    h[0]->FillRandom("gaus", ngen);

    TF1 *f1 = new TF1("f", Myfunction, -10, 10, 3);
    f1->SetParameter(0, 1);
    f1->SetParameter(1, 0);
    f1->SetParameter(2, 1);

    h[1]->FillRandom("gaus", ngen);

    TCanvas *c1 = new TCanvas("c1", "Canvas 1");
    h[0]->Draw("H");
    h[0]->Draw("E, P, SAME");
    c1->Print("test_1.pdf");

    TCanvas *c2 = new TCanvas("c2", "Canvas 2");
    h[1]->Draw("H");
    h[1]->Draw("E, P, SAME");

    TCanvas *c3 = new TCanvas("c3", "Canvas 3");
    c3->Divide(1, 2);
    for (int i = 0; i != 2; i++)
    {
        c3->cd(i + 1);
        h[i]->Draw("H");
        h[i]->Draw("E, P, SAME");
    }

    c3->Print("test.pdf");

    for (int i = 0; i != 2; i++)
    {
        std::cout << "Histogram " << i << '\n';
        std::cout << "Entries: " << h[i]->GetEntries() << '\n';
        std::cout << "Mean: " << h[i]->GetMean() << " +/- " << h[i]->GetMeanError() << '\n';
        std::cout << "RMS: " << h[i]->GetRMS() << " +/- " << h[i]->GetRMSError() << '\n';
        std::cout << "Maximum is located in bin: " << h[i]->GetMaximumBin() << '\n';
    }
}