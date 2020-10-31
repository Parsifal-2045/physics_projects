#include <iostream>
#include <vector>
#include <memory>
#include <TRandom.h>
#include <TMath.h>
#include <TH1.h>
#include <TCanvas.h>
#include <TFile.h>
#include "particle_type.hpp"
#include "resonance_type.hpp"
#include "particle.hpp"

int main()
{
    Particle::AddParticleType("Pi+", 0.13957, 1, 0);
    Particle::AddParticleType("Pi-", 0.13957, -1, 0);
    Particle::AddParticleType("K+", 0.49367, +1, 0);
    Particle::AddParticleType("K-", 0.49367, -1, 0);
    Particle::AddParticleType("P+", 0.93827, +1, 0);
    Particle::AddParticleType("P-", 0.93827, -1, 0);
    Particle::AddParticleType("K*", 0.89166, 0, 0.050);

    std::unique_ptr<TH1F> particle_type(new TH1F("particle_type", "Types of particles generated", 7, 0, 7));
    std::unique_ptr<TH1F> polar_angle(new TH1F("polar_angle", "Polar Angle", 100, 0, TMath::Pi()));
    std::unique_ptr<TH1F> azimutal_angle(new TH1F("azimutal_angle", "Azimutal Angle", 100, 0, 2 * TMath::Pi()));
    std::unique_ptr<TH1F> impulse(new TH1F("impulse", "Impulse module", 100, 0, 7));
    std::unique_ptr<TH1F> trasversal_impulse(new TH1F("trasversal_impulse", "Trasversal Impulse", 100, 0, 7));
    std::unique_ptr<TH1F> energy(new TH1F("energy", "Total Energy", 100, 0, 7));

    gRandom->SetSeed();
    int N = 120;
    Particle event[N];
    for (int i = 0; i != 1E5; i++)
    {
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
            energy->Fill(p.GetEnergy());
            int k = 0;
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
            }
            particle_type->Fill(p.GetIndexPosition());
            event[j] = p;
        }
    }

    std::unique_ptr<TCanvas> c(new TCanvas("c", "Prova"));
    c->Divide(3, 2);
    c->cd(1);
    particle_type->Draw("E, H");
    c->cd(2);
    azimutal_angle->Draw("E, H");
    c->cd(3);
    polar_angle->Draw("E, H");
    c->cd(4);
    impulse->Draw("E, H");
    c->cd(5);
    trasversal_impulse->Draw("E, H");
    c->cd(6);
    energy->Draw("E, H");

    std::unique_ptr<TFile> f(new TFile("Histograms.root", "RECREATE"));
    particle_type->Write();
    polar_angle->Write();
    azimutal_angle->Write();
    impulse->Write();
    trasversal_impulse->Write();
    energy->Write();
    c->Write();
    f->Close();

    Particle::Destructor();
}