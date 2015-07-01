#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <functional>

// Buffered file reader. Every line is a record.
struct Reader {
  FILE *fp_;

  Reader(const char* filename) {
    fp_ = fopen(filename, "r");
    if (fp_ == NULL) {
      fprintf(stderr, "Open failed: %s\n", filename);
      exit(EXIT_FAILURE);
    }
  }

  ~Reader() {
    fclose(fp_);
  }

  int Read(std::function<int(char*)> line_parser) {
    char *line = NULL;
    size_t num_byte;
    int sum = 0;
    while (getline(&line, &num_byte, fp_) != -1) {
      sum += line_parser(line);
    }
    free(line);
    return sum;
  }
};
