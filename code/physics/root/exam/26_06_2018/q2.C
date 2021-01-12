/*
Si scriva la parte rilevante ed autoconsistente del codice di una macro di ROOT in cui:

1. Si definiscono 2 istogrammi monodimensionali di 500 bin in un range da 0 a 5. 

2. Si riempe il primo istogramma con 1e6 occorrenze generate esplicitamente e 
singolarmente secondo una distribuzione gaussiana con media mu=2 e deviazione standard 
sigma=0.5.

3. Si riempe il secondo istogramma con 1e5 occorrenze generate esplicitamente e 
singolarmente secondo una distribuzione esponenziale decrescente con media mu=1.

4. Si fa la somma dei due istogrammi, e si effettua il Fit dell’istogramma somma secondo una 
forma funzionale consistente di una gaussiana (3 parametri) e un esponenziale (2 
parametri), per un totale di 5 parametri liberi.

5. Si stampa a schermo il valore dei parametri dopo il fit, con relativo errore, e il chi-quadro ridotto
*/

#include <TH1.h>
#include <TF1.h>
#include <TRandom.h>
#include <TStyle.h>

void macro2()
{
    TH1F *gaus = new TH1F("gaus", "Gaussian entries", 500, 0., 5.);
    TH1F *exp = new TH1F("exp", "Exponential entries", 500, 0., 5.);
    gRandom->SetSeed();
    for (int i = 0; i != 1e6; i++)
    {
        double value = gRandom->Gaus(2., 0.5);
        gaus->Fill(value);
    }
    for (int i = 0; i != 1e5; i++)
    {
        double value = gRandom->Exp(1);
    }
    TH1F *sum = new TH1F("sum", "Sum histogram", 500, 0., 5.);
    sum->Add(gaus, exp, 1, 1);
    TF1 *fitfunc = new TF1("fitfunc", "[0]*TMath::Gaus(x, [1], [2]) + [3]*TMath::Exp(x*[4])");
    fitfunc->SetParameters(1, sum->GetMean(), sum->GetRMS(), 1, 1);
    gStyle->SetOptFit(1111);
    sum->Fit("fitfunc");
    sum->Draw("E");
    std::cout << "Amplitude: " << fitfunc->GetParameter(0) << " +/- " << fitfunc->GetParError(0) << '\n';
    std::cout << "Mean: " << fitfunc->GetParameter(1) << " +/- " << fitfunc->GetParError(1) << '\n';
    std::cout << "Sigma: " << fitfunc->GetParameter(2) << " +/- " << fitfunc->GetParError(2) << '\n';
    std::cout << "Constant: " << fitfunc->GetParameter(3) << " +/- " << fitfunc->GetParError(3) << '\n';
    std::cout << "Exponential mean: " << fitfunc->GetParameter(4) << " +/- " << fitfunc->GetParError(4) << '\n';
    std::cout << "Chi-square / DOF: " << fitfunc->GetChisquare() << " / " << fitfunc->GetNDF() << '\n';
}