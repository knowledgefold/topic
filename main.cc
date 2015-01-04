#include "trainer.h"

#include <memory>

DEFINE_int32(mode, 1, "sample-by-doc: 1, sample-by-word: 2");
DEFINE_string(data_file, "", "Text file containing bag of words");
DEFINE_string(dump_prefix, "/tmp/dump", "Prefix for training results");

int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::ParseCommandLineFlags(&argc, &argv, false);
  FLAGS_logtostderr = true;
  FLAGS_colorlogtostderr = true;
  if (argc == 1) print_help();
  else print_flags();

  std::unique_ptr<Trainer> trainer;
  switch (FLAGS_mode) {
    case 1: trainer.reset(new Trainer1); break;
    case 2: trainer.reset(new Trainer2); break;
    default: LOG(FATAL) << "Invalid mode";
  }
  trainer->ReadData(FLAGS_data_file);
  trainer->Train();
  //trainer->SaveModel(FLAGS_dump_prefix);
}

