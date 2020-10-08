void SetStyle()
{
    gROOT->SetStyle("Plain");
    gStyle->SetPalette(57);
    gStyle->SetOptTitle(0);
}

void Maxwell()
{
    TH1F *h1 = new TH1F("h1", "Tempi di caduta", 8, -0.5, 15.5);

    ifstream in;
    in.open("maxwell.dat");
    Float_t x, y;
    while (1)
    {
        in >> x >> y;
        if (!in.good())
            break;
        for (int i = 0; i != y; i++)
        {
            h1->Fill(x);
        }
    }
    in.close();

    TCanvas *cMaxwell = new TCanvas("cMaxwell", "Tempi di caduta del pendolo di Maxwell");
    h1->GetXaxis()->SetTitle("Tempi di caduta (s)");
    h1->GetYaxis()->SetTitle("Occorrenze");
    h1->GetYaxis()->SetTitleOffset(1.3);
    h1->SetFillColor(kBlue);
    h1->SetMarkerStyle(4);
    h1->SetLineWidth(2);
    h1->SetMaximum(35);
    gStyle->SetOptStat(112210);
    h1->Draw();
    h1->Draw("E,same");

    cout << "Occorrenze totali: " << h1->GetEntries() << '\n';
    cout << "Media dell'istogramma: " << h1->GetMean() << " +/- " << h1->GetMeanError() << '\n';
    cout << "RMS: " << h1->GetRMS() << " +/- " << h1->GetRMSError() << '\n';

    //cMaxwell->Print("cMaxwell.pdf");
    //cMaxwell->Print("cMaxwell.C");
    //cMaxwell->Print("cMaxwell.root");
}