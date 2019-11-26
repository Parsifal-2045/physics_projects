#include <iostream>

int pow(int b, int e) {
  int p = 1;
  for (int i = 0; i < e; i++) {
    p = p * b;
  }
  return p;
}

int main() {
  std::cout << "Inserire la base: ";
  int b;
  std::cin >> b;
  std::cout << "Inserire l'esponente: ";
  int e;
  std::cin >> e;
  if ((b == 0) && (e == 0)) {
    std::cout << "Forma indeterminata";
  } else {
    int r = pow(b, e);
    std::cout << "Risultato: ";
    std::cout << r << '\n';
  }
}
