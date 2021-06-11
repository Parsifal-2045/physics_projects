#include <iostream>
#include <TCanvas.h>
#include <TGraph.h>
#include <TMath.h>

void graph()
{
    TCanvas *c_3 = new TCanvas("c_3", "Risposta in frequenza primo circuito");
    c_3->SetGrid();
    TGraph *g_3 = new TGraph("V_R_3.dat", "%lg %lg");
    g_3->SetTitle("Risposta in frequenza della maglia risonante a frequenza minore");
    g_3->GetXaxis()->SetTitle("Frequenza (Hz)");
    g_3->GetYaxis()->SetTitle("Ampiezza (V)");
    g_3->GetYaxis()->SetDecimals();

    auto y_3 = g_3->GetY();
    auto x_3 = g_3->GetX();
    int n_3 = g_3->GetN();
    int i_3 = std::max_element(y_3, y_3 + n_3) - y_3;
    std::cout << "Frequenza di risonanza primo circuito = " << x_3[i_3] << '\n';

    g_3->Draw();
    c_3->Print("maglia_minore.pdf");

    TCanvas *c_9 = new TCanvas("c_9", "Risposta in frequenza secondo circuito");
    c_9->SetGrid();
    TGraph *g_9 = new TGraph("V_R_9.dat", "%lg %lg");
    g_9->SetTitle("Risposta in frequenza della maglia risonante a frequenza maggiore");
    g_9->GetXaxis()->SetTitle("Frequenza (Hz)");
    g_9->GetYaxis()->SetTitle("Ampiezza (V)");
    g_9->GetYaxis()->SetDecimals();

    auto y_9 = g_9->GetY();
    auto x_9 = g_9->GetX();
    int n_9 = g_9->GetN();
    int i_9 = std::max_element(y_9, y_9 + n_9) - y_9;
    std::cout << "Frequenza di risonanza secondo circuito = " << x_9[i_9] << '\n';

    g_9->Draw();
    c_9->Print("maglia_maggiore.pdf");

    TCanvas *c_sum = new TCanvas("c_sum", "Risposta in frequenza del circuito completo");
    c_sum->SetGrid();
    TGraph *g_sum = new TGraph("V_R_tot.dat", "%lg %lg");
    g_sum->SetTitle("Risposta in frequenza del circuito completo");
    g_sum->GetXaxis()->SetTitle("Frequenza (Hz)");
    g_sum->GetYaxis()->SetTitle("Ampiezza (V)");
    g_sum->GetYaxis()->SetDecimals();

    g_sum->Draw();
    c_sum->Print("circuito_completo.pdf");
}