#include <iostream>

int isqrt (int a)
{
    int i = 1;
    while (i*i < a) {
        i++;
    } 
    if (i*i > a) {
        i--;
    }
    return i;
}

int main ()
{
    std::cout << "Inserire l'argomento: ";
    int a ;
    std::cin >> a ;
    if (a < 0) {
        std::cout << "Inserire un argomento positivo" << '\n';
    }
    else {
        int r;
        r = isqrt (a);
        std::cout << "La radice quadrata intera di " << a << " è: ";
        std:: cout << r << '\n';
    }

}