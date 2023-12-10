#include <iostream>
#include <vector>
#include <memory>
#include <TRandom.h>
#include <TMath.h>
#include <TH1.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TBenchmark.h>
#include "particle_type.hpp"
#include "resonance_type.hpp"
#include "particle.hpp"

int main()
{
    char benchmark;
    std::cout << "Do you want to benchmark the generation code? (y/n)" << '\n';
    std::cin >> benchmark;
    if (benchmark == 'y')
    {
        gBenchmark = new TBenchmark();
        gBenchmark->Start("Complete benchmark");
    }
    Particle::AddParticleType("Pi+", 0.13957, +1, 0);
    Particle::AddParticleType("Pi-", 0.13957, -1, 0);
    Particle::AddParticleType("K+", 0.49367, +1, 0);
    Particle::AddParticleType("K-", 0.49367, -1, 0);
    Particle::AddParticleType("P+", 0.93827, +1, 0);
    Particle::AddParticleType("P-", 0.93827, -1, 0);
    Particle::AddParticleType("K*", 0.89166, 0, 0.050);

    std::vector<TH1D *> histograms;
    TH1D *particle_type = new TH1D("particle_type", "Types of particles generated", 7, 0, 7);
    histograms.push_back(particle_type);
    TH1D *polar_angle = new TH1D("polar_angle", "Polar angle distribution", 100, 0, TMath::Pi());
    histograms.push_back(polar_angle);
    TH1D *azimutal_angle = new TH1D("azimutal_angle", "Azimutal angle distribution", 100, 0, 2 * TMath::Pi());
    histograms.push_back(azimutal_angle);
    TH1D *impulse = new TH1D("impulse", "Impulse module distribution", 100, 0, 7);
    histograms.push_back(impulse);
    TH1D *trasversal_impulse = new TH1D("trasversal_impulse", "Trasversal Impulse", 100, 0, 5);
    histograms.push_back(trasversal_impulse);
    TH1D *energy = new TH1D("energy", "Total Energy", 100, 0, 7);
    histograms.push_back(energy);
    TH1D *tot_inv_mass = new TH1D("tot_inv_mass", "Invariant mass", 100, 0, 2);
    histograms.push_back(tot_inv_mass);
    TH1D *disc_inv_mass = new TH1D("disc_inv_mass", "Invariant mass of particles with opposite polarity", 80, 0, 2);
    histograms.push_back(disc_inv_mass);
    TH1D *same_inv_mass = new TH1D("same_inv_mass", "Invariant mass of particles with same polarity", 80, 0, 2);
    histograms.push_back(same_inv_mass);
    TH1D *disc_pi_k = new TH1D("disc_pi_k", "Invariant mass of couples Pi, K with opposite polarity", 80, 0, 2);
    histograms.push_back(disc_pi_k);
    TH1D *same_pi_k = new TH1D("same_pi_k", "Invariant mass of couples Pi, K with same polarity", 80, 0, 2);
    histograms.push_back(same_pi_k);
    TH1D *decay_inv_mass = new TH1D("decay_inv_mass", "Invariant mass of particles generated after a K* decay", 80, 0, 2);
    histograms.push_back(decay_inv_mass);

    gRandom->SetSeed();
    int N = 120;
    Particle event[N];
    for (int i = 0; i != 1E5; i++)
    {
        int k = 0;
        for (int j = 0; j != 100; j++)
        {
            Particle p;
            double phi = gRandom->Uniform(0, 2 * TMath::Pi());
            double theta = gRandom->Uniform(0, TMath::Pi());
            double P = gRandom->Exp(1);
            double Px = P * TMath::Sin(theta) * TMath::Cos(phi);
            double Py = P * TMath::Sin(theta) * TMath::Sin(phi);
            double Pz = P * TMath::Cos(theta);
            double P_t = std::sqrt(Px * Px + Py * Py);
            p.SetP(Px, Py, Pz);
            polar_angle->Fill(theta);
            azimutal_angle->Fill(phi);
            impulse->Fill(P);
            trasversal_impulse->Fill(P_t);
            double x = gRandom->Uniform(0., 1.);
            if (x < 0.8)
            {
                double y = gRandom->Uniform(0., 1.);
                if (y < 0.5)
                {
                    p.SetAttribute("Pi+");
                }
                else
                {
                    p.SetAttribute("Pi-");
                }
            }
            else if (x > 0.8 && x < 0.9)
            {
                double y = gRandom->Uniform(0., 1.);
                if (y < 0.5)
                {
                    p.SetAttribute("K+");
                }
                else
                {
                    p.SetAttribute("K-");
                }
            }
            else if (x > 0.9 && x < 0.99)
            {
                double y = gRandom->Uniform(0., 1.);
                if (y < 0.5)
                {
                    p.SetAttribute("P+");
                }
                else
                {
                    p.SetAttribute("P-");
                }
            }
            else
            {
                p.SetAttribute("K*");
                double y = gRandom->Uniform(0., 1.);
                Particle dau1;
                Particle dau2;
                if (y < 0.5)
                {
                    dau1.SetAttribute("Pi+");
                    dau2.SetAttribute("K-");
                }
                else
                {
                    dau1.SetAttribute("Pi-");
                    dau2.SetAttribute("K+");
                }
                p.Decay2Body(dau1, dau2);
                event[100 + k] = dau1;
                k++;
                event[100 + k] = dau2;
                k++;
                decay_inv_mass->Fill(dau1.InvMass(dau2));
            }
            particle_type->Fill(p.GetIndexPosition());
            energy->Fill(p.GetEnergy());
            event[j] = p;
        }
        for (int j = 0; j != 100 + k; j++)
        {
            for (int l = j + 1; l != 100 + k; l++)
            {
                tot_inv_mass->Fill(event[l].InvMass(event[j]));
                if (event[l].GetCharge() * event[j].GetCharge() > 0)
                {
                    same_inv_mass->Fill(event[l].InvMass(event[j]));
                    if ((event[l].GetIndexPosition() == 0 && event[j].GetIndexPosition() == 2) ||
                        (event[j].GetIndexPosition() == 0 && event[l].GetIndexPosition() == 2))
                    {
                        same_pi_k->Fill(event[l].InvMass(event[j]));
                    }
                    else if ((event[l].GetIndexPosition() == 1 && event[j].GetIndexPosition() == 3) ||
                             (event[j].GetIndexPosition() == 1 && event[l].GetIndexPosition() == 3))
                    {
                        same_pi_k->Fill(event[l].InvMass(event[j]));
                    }
                }
                else if (event[l].GetCharge() * event[j].GetCharge() < 0)
                {
                    disc_inv_mass->Fill(event[l].InvMass(event[j]));
                    if ((event[l].GetIndexPosition() == 0 && event[j].GetIndexPosition() == 3) ||
                        (event[j].GetIndexPosition() == 0 && event[l].GetIndexPosition() == 3))
                    {
                        disc_pi_k->Fill(event[l].InvMass(event[j]));
                    }
                    else if ((event[l].GetIndexPosition() == 1 && event[j].GetIndexPosition() == 2) ||
                             (event[j].GetIndexPosition() == 1 && event[l].GetIndexPosition() == 2))
                    {
                        disc_pi_k->Fill(event[l].InvMass(event[j]));
                    }
                }
            }
        }
        float progress = i / 1E5;
        int barWidth = 70;
        std::cout << "[";
        int pos = barWidth * progress;
        for (int a = 0; a < barWidth; a++)
        {
            if (a < pos)
                std::cout << "=";
            else if (a == pos)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << int(progress * 100.0) << " %\r";
        std::cout.flush();
    }

    std::unique_ptr<TCanvas> c1(new TCanvas("c1", "Particle types, angles and kinetics"));
    c1->Divide(3, 2);
    c1->cd(1);
    particle_type->Draw("E, H");
    c1->cd(2);
    azimutal_angle->Draw("E, H");
    c1->cd(3);
    polar_angle->Draw("E, H");
    c1->cd(4);
    impulse->Draw("E, H");
    c1->cd(5);
    trasversal_impulse->Draw("E, H");
    c1->cd(6);
    energy->Draw("E, H");

    std::unique_ptr<TCanvas> c2(new TCanvas("c2", "Invariant mass"));
    c2->Divide(3, 2);
    c2->cd(1);
    tot_inv_mass->Draw("E, H");
    c2->cd(2);
    disc_inv_mass->Draw("E, H");
    c2->cd(3);
    same_inv_mass->Draw("E, H");
    c2->cd(4);
    disc_pi_k->Draw("E, H");
    c2->cd(5);
    same_pi_k->Draw("E, H");
    c2->cd(6);
    decay_inv_mass->Draw("E, H");

    std::unique_ptr<TFile> f(new TFile("histograms.root", "RECREATE"));
    particle_type->Write();
    polar_angle->Write();
    azimutal_angle->Write();
    impulse->Write();
    trasversal_impulse->Write();
    energy->Write();
    tot_inv_mass->Write();
    disc_inv_mass->Write();
    same_inv_mass->Write();
    disc_pi_k->Write();
    same_pi_k->Write();
    decay_inv_mass->Write();
    c1->Write();
    c2->Write();
    f->Close();

    Particle::Destructor();
    for (auto histo : histograms)
    {
        delete histo;
    }

    std::cout << std::endl;
    std::cout << "Generated 100'000 events, use analysis.C to see the results" << '\n';
    if (benchmark == 'y')
    {
        gBenchmark->Show("Complete benchmark");
    }
}