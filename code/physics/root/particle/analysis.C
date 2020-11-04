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

void SetStyle()
{
    gROOT->SetStyle("Plain");
    gStyle->SetPalette(57);
    gStyle->SetOptTitle(1);
    gStyle->SetOptStat(1110);
    gStyle->SetOptFit(111);
}

void analysis()
{
    TFile *file = new TFile("histograms.root", "READ");
    TFile *result = new TFile("analysis.root", "RECREATE");
    std::vector<TH1F *> histograms;
    TH1F *particle_type = (TH1F *)file->Get("particle_type");
    histograms.push_back(particle_type);
    particle_type->GetXaxis()->SetTitle("Particle type");
    particle_type->GetYaxis()->SetTitle("Particles generated");
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

    TCanvas *c1 = new TCanvas("c1", "Types of particle generated");
    particle_type->Draw();

    TCanvas *c2 = new TCanvas("c2", "Angles distribution");
    TH1F *polar_angle = (TH1F *)file->Get("polar_angle");
    histograms.push_back(polar_angle);
    TH1F *azimutal_angle = (TH1F *)file->Get("azimutal_angle");
    histograms.push_back(azimutal_angle);
    c2->Divide(2, 1);
    c2->cd(1);
    std::cout << red << "Fit stats for polar angle: " << reset << '\n';
    polar_angle->Fit("pol0", "B", "Q", 0, TMath::Pi());
    TF1 *f = polar_angle->GetFunction("pol0");
    std::cout << cyan << "Polar angle fit parameter (p0): " << f->GetParameter(0) << " +/- " << f->GetParError(0) << reset << '\n';
    polar_angle->Draw("E, H, SAME");
    c2->cd(2);
    std::cout << red << "Fit stats for azimutal angle: " << reset << '\n';
    azimutal_angle->Fit("pol0", "B", "SAME, Q", 0, 2 * TMath::Pi());
    f = azimutal_angle->GetFunction("pol0");
    std::cout << cyan << "Azimutal angle fit parameter (p0): " << f->GetParameter(0) << " +/- " << f->GetParError(0) << reset << '\n';
    azimutal_angle->Draw("E, H, SAME");

    TCanvas *c3 = new TCanvas("c3", "Impulse");
    TH1F *impulse = (TH1F *)file->Get("impulse");
    histograms.push_back(impulse);
    std::cout << red << "Fit stats for impulse: " << reset << '\n';
    impulse->Fit("expo");
    impulse->Draw("E, H, SAME");
    std::cout << cyan << "Impulse mean: " << impulse->GetMean() << " + / - " << impulse->GetMeanError() << reset << '\n';

    std::vector<TH1F *> hist_inv_mass;
    TH1F *disc_inv_mass = (TH1F *)file->Get("disc_inv_mass");
    hist_inv_mass.push_back(disc_inv_mass);
    TH1F *same_inv_mass = (TH1F *)file->Get("same_inv_mass");
    hist_inv_mass.push_back(same_inv_mass);
    TH1F *disc_pi_k = (TH1F *)file->Get("disc_pi_k");
    hist_inv_mass.push_back(disc_pi_k);
    TH1F *same_pi_k = (TH1F *)file->Get("same_pi_k");
    hist_inv_mass.push_back(same_pi_k);
    TH1F *decay_inv_mass = (TH1F *)file->Get("decay_inv_mass");
    hist_inv_mass.push_back(decay_inv_mass);

    TCanvas *c4 = new TCanvas("c4", "Invariant mass difference");
    c4->Divide(1, 2);
    c4->cd(1);
    TH1F *diff_disc_same_pi_k = new TH1F("diff_disc_same_pi_k", "Difference between Pions and Kaons of opposite and same polarities", 80, 0, 2);
    hist_inv_mass.push_back(diff_disc_same_pi_k);
    diff_disc_same_pi_k->Add(hist_inv_mass[2], hist_inv_mass[3], 1, -1);
    diff_disc_same_pi_k->GetXaxis()->SetTitle("Mass [GeV]");
    diff_disc_same_pi_k->GetYaxis()->SetTitle("Entries");

    std::cout << red << "Fit stats for invariant mass of Pions and Kaons: " << reset << '\n';
    diff_disc_same_pi_k->Fit("gaus");
    diff_disc_same_pi_k->Draw("E, H, SAME");
    f = diff_disc_same_pi_k->GetFunction("gaus");
    std::cout << cyan << "Mean (K* mass): " << f->GetParameter(1) << " +/- " << f->GetParError(1) << '\n';
    std::cout << cyan << "Sigma (K* width): " << f->GetParameter(2) << " +/- " << f->GetParError(2) << reset << '\n';

    c4->cd(2);
    TH1F *diff_disc_same_tot = new TH1F("diff_disc_same_tot", "Difference betweeen all particles of opposite and same polarities", 80, 0, 2);
    hist_inv_mass.push_back(diff_disc_same_tot);
    diff_disc_same_tot->Add(disc_inv_mass, same_inv_mass, 1, -1);
    diff_disc_same_tot->GetXaxis()->SetTitle("Mass [GeV]");
    diff_disc_same_tot->GetYaxis()->SetTitle("Entries");

    std::cout << red << "Fit stats for total invariant mass: " << reset << '\n';
    diff_disc_same_tot->Fit("gaus");
    diff_disc_same_tot->Draw("E, H, SAME");
    f = diff_disc_same_tot->GetFunction("gaus");
    std::cout << cyan << "Mean (K* mass): " << f->GetParameter(1) << " +/- " << f->GetParError(1) << '\n';
    std::cout << cyan << "Sigma (K* width): " << f->GetParameter(2) << " +/- " << f->GetParError(2) << reset << '\n';

    result->Write();
}