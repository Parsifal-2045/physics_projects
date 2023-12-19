#include "TROOT.h"

void compile()
{
    gROOT->LoadMacro("hit.cxx++");
    gROOT->LoadMacro("trueTrack.cxx++");
    gROOT->LoadMacro("recoTrack.cxx++");
    gROOT->LoadMacro("main.cxx++");
}
