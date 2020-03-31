#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
  int lines = atoi(argv[1]); // the first argument represents length and width
  for (int i = 0; i < lines; i++)
  {
    for (int j = 0; j < lines; j++)
    {
      cout << i*lines + j << " ";
    }
    cout << endl;
  }
  cout << endl;
}