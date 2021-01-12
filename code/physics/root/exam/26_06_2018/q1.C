/*
Si scriva la parte rilevante ed autoconsistente del codice di una macro di ROOT in cui:

1. Si definiscono 2 istogrammi monodimensionali di 1000 bin in un range da 0 a 10. 

2. Si riempe il primo istogramma con 107
occorrenze di una variabile casuale x generate 
esplicitamente e singolarmente e distribuite secondo una distribuzione gaussiana
con mu=5 e deviazione standard sigma=1 (campione totale).

3. Su tali occorrenze, si simula (attraverso un criterio di reiezione di tipo “hit or miss”) 
un’efficienza di rivelazione dipendente dalla variabile casuale x secondo la forma:
epsilon(x) = 0.1 * x * e^-x
Riempire il secondo istogramma con le occorrenze accettate (campione accettato).

4. Si effettua la divisione fra i due istogrammi per ottenere l’efficienza di rivelazione 
osservata, utilizzando il metodo Divide della classe degli istogrammi e inserendo 
l’opportuna opzione per la valutazione degli errori secondo la statistica binomiale. 

5. Si disegna l’istogramma dell’ efficienza visualizzando le incertezze sui contenuti dei bin. 
*/
#include <TH1.h>
#include <TF1.h>
#include <TRandom.h>

void macro1()
{
    TH1F *gen = new TH1F("gen", "All entries", 1000, 0., 10.);
    TH1F *acc = new TH1F("acc", "Accepted entries", 1000, 0., 10.);
    TF1 *epsilon = new TF1("epsilon", "0.1*x*exp(-x)");
    gRandom->SetSeed();
    for (int i = 0; i != 1e7; i++)
    {
        double x = gRandom->Gaus(5., 1.);
        gen->Fill(x);
        auto y = gRandom->Uniform(0., 1.);
        if (y < epsilon->Eval(x))
        {
            acc->Fill(x);
        }
    }
    TH1F *eff = new TH1F("eff", "Efficiency histogram", 1000, 0., 10.);
    eff->Divide(acc, gen, 1, 1, "B");
    eff->Draw("E");
}