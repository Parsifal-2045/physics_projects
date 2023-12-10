/*
Si scriva la parte rilevante ed autoconsistente del codice di una macro di ROOT in cui:

1. Si definisce un istogramma monodimensionale di 100 bin in un range da 0 a 10. 

2. Si riempe l’istogramma con 1e5 occorrenze di una variabile casuale x distribuita secondo la 
p.d.f. f(x)=sin(x)+x^2 nel range [0,10] , utilizzando il metodo FillRandom(const char* f,Int_t 
N) della classe di istogrammi.
*/

#include <TH1.h>
#include <TF1.h>

void macro3(int ngen = 1e5)
{
    TH1F *h = new TH1F("h", "histogram", 100, 0., 10.);
    TF1 *f = new TF1("f", "sin(x) + (x*x)", 0., 10.);
    h->FillRandom("f", ngen);
    h->Draw();
}