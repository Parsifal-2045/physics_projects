void GraphEvolution()
{
    TCanvas *canvas = new TCanvas("canvas", "Andamento epidemiologico");
    canvas->SetGrid();

    TGraph *S = new TGraph("dati_S.dat", "%lg %lg");
    S->SetTitle("Andamento epidemiologico");
    S->SetLineColor(kBlue);
    S->GetXaxis()->SetTitle("Tempo (giorni)");
    S->GetYaxis()->SetTitle("Numero di persone");
    S->GetYaxis()->SetTitleOffset(1.5);

    TGraph *I = new TGraph("dati_I.dat", "%lg %lg");
    I->SetTitle("Andamento infetti");
    I->SetLineColor(kRed);

    TGraph *R = new TGraph("dati_R.dat", "%lg %lg");
    R->SetTitle("Andamento guariti");
    R->SetLineColor(kGreen);

    TGraph *D = new TGraph("dati_D.dat", "%lg %lg");
    D->SetTitle("Andamento decessi");
    D->SetLineColor(kBlack);

    S->Draw("AC");
    I->Draw("C, SAME");
    R->Draw("C, SAME");
    D->Draw("C, SAME");

    TLegend *leg = new TLegend(.75, .70, .99, .99);
    leg->SetFillColor(0);
    leg->AddEntry(S, "Persone suscettibili");
    leg->AddEntry(I, "Persone infette");
    leg->AddEntry(R, "Persone guarite");
    leg->AddEntry(D, "Persone decedute");
    leg->SetEntrySeparation(0.05);
    leg->Draw("SAME");

    canvas->Print("Andamento epidemiologico.pdf");
}