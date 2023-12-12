#ifndef HIT_H
#define HIT_H

class hit
{
public:
    hit();
    hit(double x, double y, int trueTrackIndex);
    double GetX() const;
    double GetY() const;
    double GetTrackIndex() const;
    void Set(double x, double y, int trueTrackIndex);

private:
    double fX, fY;                         // x-y position
    int fTrackIndex;                       // the label of the generated track that created the hit
    static constexpr double fgkYRes = 0.2; // detector spatial resolution in y (cm)
};
#endif
