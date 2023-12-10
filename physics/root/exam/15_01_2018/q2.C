/*
Quesito 2 (12 punti)
Si scriva la parte rilevante ed autoconsistente del codice di una macro di ROOT in cui:
1. Si definiscono 2 istogrammi monodimensionali di 500 bin in un range da 0 a 5. 
2. Si riempe il primo istogramma con 1e6 occorrenze generate esplicitamente e 
singolarmente secondo una distribuzione gaussiana con media mu=2.5 e deviazione 
standard sigma=0.25.
3. Si riempe il secondo istogramma con 1e4 occorrenze generate esplicitamente e 
singolarmente secondo una distribuzione uniforme del range [0,5]
4. Si fa la somma dei due istogrammi, e si effettua il Fit dell’istogramma somma secondo una 
forma funzionale consistente di una gaussiana (3 parametri: ampiezza,media e deviazione 
standard) e un polinomio di grado 0 (1 parametro), per un totale di 4 parametri liberi.
5. Si stampa a schermo il valore dei parametri dopo il fit, con relativo errore, e il chi-quadro ridotto
*/

#include <TH1.h>
#include <TRandom.h>
#include <TF1.h>
#include <TStyle.h>

void macro2()
{
    gStyle->SetOptFit(1111);
    TH1F *gaus = new TH1F("gaus", "gaussian entries", 500, 0., 5.);
    TH1F *uni = new TH1F("uni", "uniform entries", 500, 0., 5.);
    for (int i = 0; i != 1e6; i++)
    {
        auto value = gRandom->Gaus(2.5, 0.25);
        gaus->Fill(value);
    }
    for (int i = 0; i != 1e4; i++)
    {
        auto value = gRandom->Uniform(0., 5.);
        uni->Fill(value);
    }
    TH1F *sum = new TH1F("sum", "sum histogram", 500, 0., 5.);
    sum->Add(gaus, uni, 1, 1);
    TF1 *fitfunc = new TF1("fitfunc", "[0]*TMath::Gaus(x, [1], [2]) + [3]");
    fitfunc->SetParameters(1, sum->GetMean(), sum->GetRMS(), 1);
    sum->Fit("fitfunc");
    std::cout << "Fit parameters:" << '\n';
    std::cout << "Amplitude: " << fitfunc->GetParameter(0) << " +/- " << fitfunc->GetParError(0) <<'\n';
    std::cout << "Mean: " << fitfunc->GetParameter(1) << " +/- " << fitfunc->GetParError(1) <<'\n';
    std::cout << "Sigma: " << fitfunc->GetParameter(2) << " +/- " << fitfunc->GetParError(2) <<'\n';
    std::cout << "Constant: " << fitfunc->GetParameter(3) << " +/- " << fitfunc->GetParError(3) <<'\n';
    std::cout << "Chi-square / DOF: " << fitfunc->GetChisquare() << " / " << fitfunc->GetNDF() << '\n';
    sum->Draw("E, SAME");
}