
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

int mcm(int a, int b) // Funzione minimo comune multiplo
{
    if (a == b)
    {
        return a;
    }
    if (gcd(a, b) == 1)
    {
        int m = a * b;
        return m;
    }
    if (gcd(a, b) != 1)
    {
        int m = (a * b) / gcd(a, b);
        return m;
    }
}