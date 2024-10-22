
int *factory() { return new int; }
// "still reachable"
auto g = factory();

int main()
{
  // "definitely lost"
  auto t = factory();
  // try to use both valgrind e Asan (-fsanitize=address), what happens?
  //  delete t;
  //  delete g;
}
