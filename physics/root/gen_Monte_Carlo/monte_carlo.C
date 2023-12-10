void GaussianSum()
{
    TH1F *h1 = new TH1F
    Double_t x1, x2, x3;
    for (int i = 0; i < 1000000; i++)
    {
        x1 = gRandom->Gaus(-1, 3);
        x2 = gRandom->Gaus(1, 4);
        x3 = x1 + x2;
        h1->Fill(x1);
        h2->Fill(x2);
        h3->Fill(x3);
    }
}