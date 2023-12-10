void SetStyle()
{
    gROOT->SetStyle("Plain");
    gStyle->SetPalette(57);
    gStyle->SetOptTitle(0);
}

void MeanQuadraticDistance()
{
    TCanvas *canvas = new TCanvas("canvas", "Dipendenza del cammino quadratico medio dal tempo");
    canvas->SetGrid();

    TGraph *L = new TGraph("mean_free_path.dat", "%lg %lg");
    L->SetTitle("Cammino quadratico medio");
    L->GetXaxis()->SetTitle("Tempo (s)");
    L->GetYaxis()->SetTitle("Cammino quadratico medio (um^2)");

    L->SetMarkerStyle(kOpenCircle);
    L->SetMarkerColor(kBlue);
    L->SetLineColor(kBlue);

    TF1 *f = new TF1("Linear Law", "x*[0]", 0., 10.);
    f->SetLineColor(kRed);
    f->SetLineStyle(2);

    L->Fit(f);

    gStyle->SetOptFit(111);

    L->Draw("APE");

    TLegend *leg = new TLegend(.1,.7,.3,.9, "Cammino quadratico in funzione del tempo");
    leg->SetFillColor(0);
    leg->AddEntry(L, "Cammino quadratico medio");
    leg->AddEntry(f, "Fit lineare ad un parametro");
    leg->Draw("SAME");

    cout << "Linear correlation coefficient = " << L->GetCorrelationFactor() << '\n';
    cout << "Chisquare of the fit = " << L->Chisquare(f) << '\n';
    cout << "Number of d.o.f. = " << f->GetNDF() << '\n';
    cout << "Chisquare per d.o.f. = " << f->GetChisquare() / f->GetNDF() << '\n';

    canvas->Print("Mean Free Path.pdf");
}

void StepLength()
{
    TCanvas *canvas = new TCanvas();
    canvas->SetTitle("Conteggi delle distanze di salto");

    TH1F *sl = new TH1F("sl", "Conteggi delle distanze di salto", 22, -4., 4.);

    ifstream in;
    in.open("step_length.dat");
    Float_t x, y;
    while (1)
    {
        in >> x >> y;
        if (!in.good())
        {
            break;
        }
        for (int i = 0; i != y; i++)
        {
            sl->Fill(x);
        }
    }
    in.close();

    sl->GetXaxis()->SetTitle("Distanze di salto (um/s)");
    sl->GetYaxis()->SetTitle("Conteggi");
    sl->GetYaxis()->SetTitleOffset(1.3);
    sl->SetFillColor(kRed);
    sl->SetMarkerStyle(4);
    sl->SetLineWidth(2);
    sl->SetLineColor(kBlack);
    gStyle->SetOptStat(2210);
    sl->Fit("gaus");
    sl->GetFunction("gaus")->SetLineColor(kBlack);
    gStyle->SetOptFit(111);
    sl->Draw();
    sl->Draw("E,SAME");

    cout << "Total entries : " << sl->GetEntries() << '\n';
    cout << "Mean value : " << sl->GetMean() << " +/- " << sl->GetMeanError() << '\n';
    cout << "RMS : " << sl->GetRMS() << " +/- " << sl->GetRMSError() << '\n';
    cout << "Variance : " << sl->GetRMS()*sl->GetRMS() << " +/- " << 2*sl->GetRMS()*sl->GetRMSError() << '\n';

    canvas->Print("Step Length.pdf");
}