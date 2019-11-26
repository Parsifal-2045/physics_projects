

int gcd(int a, int b) // Funzione massimo comun divisore
{
    while (a != b)
    {
        if (a > b)
        {
            a -= b; //come scrivere a = a - b
        }
        else
        {
            b -= a; //come scrivere b = b - a
        }
    }
    return a;
}