#include "flag.h"
#include "trainer.h"

int main(int argc, char** argv) {
  flag.Parse(argc, argv);
  flag.Print();

  Trainer trainer;
  trainer.Train();
  
  return 0;
}
