void SetStyle()
{
    gROOT->SetStyle("Plain");
    gStyle->SetPalette(57);
    gStyle->SetOptTitle(0);
}

void maxwell()
{
    TH1F *h1 = new TH1F("h1", "Tempi di caduta campione 1", 10, -0.5, 15.5);

    ifstream in1;
    in1.open("histo1.dat");
    Float_t x1, y1;
    while (1)
    {
        in1 >> x1 >> y1;
        if (!in1.good())
            break;
        h1->Fill(y1);
    }
    in1.close();

    TH1F *h2 = new TH1F("h2", "Tempi di caduta campione 2", 10, -0.5, 15.5);

    ifstream in2;
    in2.open("histo2.dat");
    Float_t x2, y2;
    while (1)
    {
        in2 >> x2 >> y2;
        if (!in2.good())
            break;
        h2->Fill(y2);
    }
    in2.close();

    TH1F *hRatio = new TH1F(*h1);
    hRatio->Divide(h1, h2);

    TH1F *hSum = new TH1F(*h1);
    hSum->Add(h1, h2);

    TCanvas *dati = new TCanvas("dati", "Dati campione 1 e 2");
    h1->GetXaxis()->SetTitle("Tempi di caduta (s)");
    h1->GetYaxis()->SetTitle("Occorrenze");
    h1->GetYaxis()->SetTitleOffset(1.);
    h1->SetMarkerStyle(4);
    h1->SetLineWidth(2);
    h1->SetMaximum(35);
    h1->SetFillColor(kBlue);
    h2->GetXaxis()->SetTitle("Tempi di caduta (s)");
    h2->GetYaxis()->SetTitle("Occorrenze");
    h2->GetYaxis()->SetTitleOffset(1.);
    h2->SetMarkerStyle(4);
    h2->SetLineWidth(2);
    h2->SetMaximum(35);
    h2->SetFillColor(kRed);
    gStyle->SetOptStat(112210);
    h1->Draw();
    h2->Draw("SAME");

    TCanvas *rapporto = new TCanvas("rapporto", "Rapporto dati campione 1 e 2");
    hRatio->SetLineWidth(1);
    hRatio->SetMarkerStyle(4);
    hRatio->SetFillColor(kYellow);
    hRatio->SetMaximum(5);
    gStyle->SetOptStat(112210);
    hRatio->Draw();
    hRatio->Draw("E,SAME");

    TCanvas *sum = new TCanvas("sum", "Somma dati campione 1 e 2");
    hSum->GetXaxis()->SetTitle("Tempi di caduta (s)");
    hSum->GetYaxis()->SetTitle("Occorrenze");
    hSum->GetYaxis()->SetTitleOffset(1.);
    hSum->SetMarkerStyle(4);
    hSum->SetLineWidth(1);
    hSum->SetFillColor(kBlue);
    hSum->SetMaximum(50);
    gStyle->SetOptStat(112210);
    hSum->Draw();
    hSum->Draw("E,SAME");

    cout << "Mean of the sum: " << hSum->GetMean() << " +/- " << hSum->GetMeanError() << '\n';
    cout << "RMS of the sum: " << hSum->GetRMS() << " +/- " << hSum->GetRMSError() << '\n';
}