#include <iostream>

double det(double a)
{
    double r = a;
    return r;
}

double det2(double a, double b, double c, double d)
{
    double r = (a*d) - (b*c);
    return r;
}

double det3(double a, double b, double c, double d, double e, double f, double g, double h, double i)
{
    double r = (a*e*i) + (b*f*g) + (c*d*h) - (g*e*c) - (h*f*a) - (i*d*b);
    return r;
}

double det4(double a, double b, double c, double d, double e, double f, double g, double h, double i, double j, double k, double l, double m, double n, double o, double p)
{
    double r = (a * det3(f, g, h, j, k, l, n, o, p)) - (b * det3(e, g, h, i, k, l, m, o, p)) + (c * det3(e, f, h, i, j, l, m, n, p)) -(d * det3(e, f, g, i, j, k, m, n, o));
    return r;
}

int main()
{
    std::cout << "Dichiarare l'ordine della matrice: ";
    int i;
    std::cin >> i;
    if (i == 1) {
    std::cout << "Inserire l'elemento a11: ";
    double a;
    std::cin >> a;
    std::cout << "Il determinante della matrice è: " << a << '\n';
    }
    if (i == 2) {
        std:: cout << "Inserire l'elemento a11: ";
        double a;
        std::cin >> a;
        std:: cout << "Inserire l'elemento a12: ";
        double b;
        std::cin >> b;
        std:: cout << "Inserire l'elemento a21: ";
        double c;
        std::cin >> c;
        std:: cout << "Inserire l'elemento a22: ";
        double d;
        std::cin >> d;
        double r = det2(a, b, c, d);
        std::cout << "Il determinante della matrice è: " << r << '\n';
    }
    if (i == 3) {
        std:: cout << "Inserire l'elemento a11: ";
        double a;
        std::cin >> a;
        std:: cout << "Inserire l'elemento a12: ";
        double b;
        std::cin >> b;
        std:: cout << "Inserire l'elemento a13: ";
        double c;
        std::cin >> c;
        std:: cout << "Inserire l'elemento a21: ";
        double d;
        std::cin >> d;
        std:: cout << "Inserire l'elemento a22: ";
        double e;
        std::cin >> e;
        std:: cout << "Inserire l'elemento a23: ";
        double f;
        std::cin >> f;
        std:: cout << "Inserire l'elemento a31: ";
        double g;
        std::cin >> g;
        std:: cout << "Inserire l'elemento a32: ";
        double h;
        std::cin >> h;
        std:: cout << "Inserire l'elemento a33: ";
        double i;
        std::cin >> i;
        double r = det3(a, b, c, d, e, f, g, h, i);
        std:: cout << "Il determinante della matrice è: " << r << '\n';
    }
    if (i == 4) {
        std:: cout << "Inserire l'elemento a11: ";
        double a;
        std::cin >> a;
        std:: cout << "Inserire l'elemento a12: ";
        double b;
        std::cin >> b;
        std:: cout << "Inserire l'elemento a13: ";
        double c;
        std::cin >> c;
        std:: cout << "Inserire l'elemento a14: ";
        double d;
        std::cin >> d;
        std:: cout << "Inserire l'elemento a21: ";
        double e;
        std::cin >> e;
        std:: cout << "Inserire l'elemento a22: ";
        double f;
        std::cin >> f;
        std:: cout << "Inserire l'elemento a23: ";
        double g;
        std::cin >> g;
        std:: cout << "Inserire l'elemento a24: ";
        double h;
        std::cin >> h;
        std:: cout << "Inserire l'elemento a31: ";
        double i;
        std::cin >> i;
        std:: cout << "Inserire l'elemento a32: ";
        double j;
        std::cin >> j;
        std:: cout << "Inserire l'elemento a33: ";
        double k;
        std::cin >> k;
        std:: cout << "Inserire l'elemento a34: ";
        double l;
        std::cin >> l;
        std:: cout << "Inserire l'elemento a41: ";
        double m;
        std::cin >> m;
        std:: cout << "Inserire l'elemento a42: ";
        double n;
        std::cin >> n;
        std:: cout << "Inserire l'elemento a43: ";
        double o;
        std::cin >> o;
        std:: cout << "Inserire l'elemento a44: ";
        double p;
        std::cin >> p;
        double r = det4(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
        std::cout << "Il determinante della matrice è: " << r << '\n';   
    }
    if (i != 1 && i != 2 && i != 3 && i != 4) {
        std::cout << "Il rango della matrice deve essere un intero compreso tra 1 e 4" << '\n';
    }
}