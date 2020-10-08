void histo(Int_t nev = 1.E3, Float_t mean = 0., Float_t width = 1.)
{
    TH1F *h = new TH1F("h", "x distribution", 100, -5., 5.);
    Float_t x = 0;
    for (Int_t i = 0; i < nev; i++)
    {
        x = gRandom->Gaus(mean, width);
        h->Fill(x);
    }
    h->Draw("E");
    h->Draw("HISTO, SAME");

    cout << "Mean = " << h->GetMean() << " +/- " << h->GetMeanError() << '\n';
    cout << "RMS = " << h->GetRMS() << " +/- " << h->GetRMSError() << '\n';
    cout << "Underflows = " << h->GetBinContent(0) << ", Overflows = " << h->GetBinContent(101) << '\n';
    cout << "Maximum = " << h->GetMaximum() << " Location of maximum = " << h->GetBinCenter(h->GetMaximumBin()) << '\n'; 
    TFile *file = new TFile("example.root", "RECREATE");
    h->Write();
    file->Close();
}