#include <iostream>
#include <string>
#include <TROOT.h>
#include <TStyle.h>
#include <TFile.h>
#include <TH1.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TMath.h>
#include <TMinuit.h>
#include <vector>

void analysis()
{
    TFile *file = new TFile("histograms.root", "READ");
    TH1F *histograms[7];
    TH1F *particle_type = (TH1F *)file->Get("particle_type");
    histograms[0] = particle_type;
    particle_type->GetXaxis()->SetTitle("Particle type");
    particle_type->GetXaxis()->SetBinLabel(1, "Pi+");
    particle_type->GetXaxis()->SetBinLabel(2, "Pi-");
    particle_type->GetXaxis()->SetBinLabel(3, "K+");
    particle_type->GetXaxis()->SetBinLabel(4, "K-");
    particle_type->GetXaxis()->SetBinLabel(5, "P+");
    particle_type->GetXaxis()->SetBinLabel(6, "P-");
    particle_type->GetXaxis()->SetBinLabel(7, "K*");
    const std::string red("\033[0;31m");
    const std::string cyan("\033[0;36m");
    const std::string reset("\033[0m");
    std::cout << "\033c";
    std::cout << red << "Particles generated: " << reset << particle_type->GetEntries() << '\n';
    double value = particle_type->GetBinContent(1) + particle_type->GetBinContent(2);
    double error = particle_type->GetBinError(1) + particle_type->GetBinError(2);
    double percentage = 100 * value / particle_type->GetEntries();
    std::cout << "Pions: " << value << " +/- " << error << " ~ " << percentage << "% of total" << '\n';
    value = particle_type->GetBinContent(3) + particle_type->GetBinContent(4);
    error = particle_type->GetBinError(3) + particle_type->GetBinError(4);
    percentage = 100 * value / particle_type->GetEntries();
    std::cout << "Kaons: " << value << " +/- " << error << " ~ " << percentage << "% of total" << '\n';
    value = particle_type->GetBinContent(5) + particle_type->GetBinContent(6);
    error = particle_type->GetBinError(5) + particle_type->GetBinError(6);
    percentage = 100 * value / particle_type->GetEntries();
    std::cout << "Protons: " << value << " +/- " << error << " ~ " << percentage << "% of total" << '\n';
    value = particle_type->GetBinContent(7);
    error = particle_type->GetBinError(7);
    percentage = 100 * value / particle_type->GetEntries();
    std::cout << "K* (resonance): " << value << " +/- " << error << " ~ " << percentage << "% of total" << '\n';

    TH1F *polar_angle = (TH1F *)file->Get("polar_angle");
    histograms[1] = polar_angle;
    polar_angle->GetXaxis()->SetTitle("Angle");
    std::cout << red << "Fit stats for polar angle: " << reset << '\n';
    polar_angle->Fit("pol0", "B", "Q", 0, TMath::Pi());
    TF1 *f = polar_angle->GetFunction("pol0");
    f->SetLineColor(kRed);
    std::cout << cyan << "Polar angle fit parameter (p0): " << f->GetParameter(0) << " +/- " << f->GetParError(0) << reset << '\n';
    std::cout << red << "Fit stats for azimutal angle: " << reset << '\n';

    TH1F *azimutal_angle = (TH1F *)file->Get("azimutal_angle");
    histograms[2] = azimutal_angle;
    azimutal_angle->GetXaxis()->SetTitle("Angle");
    azimutal_angle->Fit("pol0", "B", "SAME, Q", 0, 2 * TMath::Pi());
    f = azimutal_angle->GetFunction("pol0");
    std::cout << cyan << "Azimutal angle fit parameter (p0): " << f->GetParameter(0) << " +/- " << f->GetParError(0) << reset << '\n';

    TH1F *impulse = (TH1F *)file->Get("impulse");
    histograms[3] = impulse;
    impulse->GetXaxis()->SetTitle("Impulse module [GeV]");
    std::cout << red << "Fit stats for impulse: " << reset << '\n';
    impulse->Fit("expo");
    f = impulse->GetFunction("expo");
    std::cout << cyan << "Impulse fit parameters: \n";
    std::cout << "Slope: " << f->GetParameter(1) << " +/- " << f->GetParError(1) << '\n';
    std::cout << "Constant: " << f->GetParameter(0) << " +/- " << f->GetParError(0) << '\n';
    std::cout << cyan << "Impulse mean: " << impulse->GetMean() << " + / - " << impulse->GetMeanError() << reset << '\n';

    TH1F *decay_inv_mass = (TH1F *)file->Get("decay_inv_mass");
    histograms[4] = decay_inv_mass;
    decay_inv_mass->GetXaxis()->SetTitle("Mass [GeV]");
    std::cout << red << "Fit stats for invariant mass only of products of the decay (control): " << reset << '\n';
    decay_inv_mass->Fit("gaus");
    decay_inv_mass->Draw("E, H, SAME");
    f = decay_inv_mass->GetFunction("gaus");
    std::cout << cyan << "Mean (K* mass): " << f->GetParameter(1) << " +/- " << f->GetParError(1) << '\n';
    std::cout << cyan << "Sigma (K* width): " << f->GetParameter(2) << " +/- " << f->GetParError(2) << reset << '\n';

    TH1F *disc_inv_mass = (TH1F *)file->Get("disc_inv_mass");
    TH1F *same_inv_mass = (TH1F *)file->Get("same_inv_mass");
    TH1F *diff_disc_same_tot = new TH1F("diff_disc_same_tot", "Difference betweeen all particles of opposite and same polarities", 80, 0, 2);
    histograms[5] = diff_disc_same_tot;
    diff_disc_same_tot->Add(disc_inv_mass, same_inv_mass, 1, -1);
    diff_disc_same_tot->GetXaxis()->SetTitle("Mass [GeV]");
    std::cout << red << "Fit stats for total invariant mass: " << reset << '\n';
    diff_disc_same_tot->Fit("gaus");
    diff_disc_same_tot->Draw("E, H, SAME");
    f = diff_disc_same_tot->GetFunction("gaus");
    std::cout << cyan << "Mean (K* mass): " << f->GetParameter(1) << " +/- " << f->GetParError(1) << '\n';
    std::cout << cyan << "Sigma (K* width): " << f->GetParameter(2) << " +/- " << f->GetParError(2) << reset << '\n';

    TH1F *disc_pi_k = (TH1F *)file->Get("disc_pi_k");
    TH1F *same_pi_k = (TH1F *)file->Get("same_pi_k");
    TH1F *diff_disc_same_pi_k = new TH1F("diff_disc_same_pi_k", "Difference between Pions and Kaons of opposite and same polarities", 80, 0, 2);
    histograms[6] = diff_disc_same_pi_k;
    diff_disc_same_pi_k->Add(disc_pi_k, same_pi_k, 1, -1);
    diff_disc_same_pi_k->GetXaxis()->SetTitle("Mass [GeV]");
    std::cout << red << "Fit stats for invariant mass of Pions and Kaons: " << reset << '\n';
    diff_disc_same_pi_k->Fit("gaus");
    diff_disc_same_pi_k->Draw("E, H, SAME");
    f = diff_disc_same_pi_k->GetFunction("gaus");
    std::cout << cyan << "Mean (K* mass): " << f->GetParameter(1) << " +/- " << f->GetParError(1) << '\n';
    std::cout << cyan << "Sigma (K* width): " << f->GetParameter(2) << " +/- " << f->GetParError(2) << reset << '\n';

    gStyle->SetOptTitle(1);
    gStyle->SetOptStat(1110);
    gStyle->SetOptFit(111);
    for (int i = 0; i != 7; i++)
    {
        histograms[i]->GetYaxis()->SetTitle("Entries");
        histograms[i]->SetFillColor(kBlue);
        histograms[i]->SetLineColor(kBlack);
    }

    TCanvas *c1 = new TCanvas("c1", "Particles generated, impulse and angles");
    c1->Divide(2, 2);
    c1->cd(1);
    particle_type->Draw("E, H, SAME");
    c1->cd(2);
    impulse->Draw("E, H, SAME");
    c1->cd(3);
    polar_angle->Draw("E, H, SAME");
    c1->cd(4);
    azimutal_angle->Draw("E, H, SAME");

    TCanvas *c2 = new TCanvas("c2", "Invariant mass difference");
    c2->Divide(1, 3);
    c2->cd(1);
    decay_inv_mass->Draw("E, H, SAME");
    c2->cd(2);
    diff_disc_same_tot->Draw("E, H, SAME");
    c2->cd(3);
    diff_disc_same_pi_k->Draw("E, H, SAME");

    TFile *result = new TFile("analysis_result.root", "RECREATE");
    for (int i = 0; i != 7; i++)
    {
        histograms[i]->Write();
    }
    c1->Write();
    c2->Write();
}