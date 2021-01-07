/*
Quesito 1 (10 punti)
 Si scriva la parte rilevante ed autoconsistente del codice di una macro di ROOT in cui:
1. Si definiscono 2 istogrammi monodimensionali di 1000 bin in un range da 0 a 5. 
2. Si riempe il primo istogramma con 1e7 occorrenze di una variabile casuale x generate 
esplicitamente e singolarmente (i.e. attraverso gRandom) e distribuite secondo una 
distribuzione esponenziale decrescente con media mu=1 (campione totale).
3. Su tali occorrenze, si simula (attraverso un criterio di reiezione di tipo “hit or miss”) 
un’efficienza di rivelazione dipendente dalla variabile casuale x secondo la forma:
    epsilon(x)=x/5. 
Riempire il secondo istogramma con le occorrenze accettate (campione accettato).
4. Si effettua la divisione fra i due istogrammi per ottenere l’efficienza di rivelazione 
osservata, utilizzando il metodo Divide della classe degli istogrammi e inserendo 
l’opportuna opzione per la valutazione degli errori secondo la statistica binomiale. 
5. Si disegna l’istogramma dell’ efficienza visualizzando le incertezze sui contenuti dei bin. 
*/

#include <TRandom.h>
#include <TF1.h>
#include <TH1.h>

void macro1()
{
    TH1F *hacc = new TH1F("hacc", "accepted entries", 1000., 0., 5.);
    TH1F *hgen = new TH1F("hgen", "all entries", 1000., 0., 5.);
    TF1 *eff = new TF1("eff", "x/5");
    gRandom->SetSeed();
    for (int i = 0; i != 1e7; i++)
    {
        auto x = gRandom->Exp(1);
        hgen->Fill(x);
        auto y = gRandom->Uniform(0., 1.);
        if (y < eff->Eval(x))
        {
            hacc->Fill(x);
        }
    }
    TH1F *heff = new TH1F("heff", "efficiency histogram", 1000., 0., 5.);
    heff->Divide(hacc, hgen, 1, 1, "B");
    heff->Draw("E");
}