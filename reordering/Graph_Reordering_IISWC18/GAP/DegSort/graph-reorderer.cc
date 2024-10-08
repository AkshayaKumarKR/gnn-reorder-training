// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <iostream>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "reader.h"
#include "writer.h"

using namespace std;

int main(int argc, char* argv[]) {
  CLConvert cli(argc, argv, "converter");
  cli.ParseArgs();
  if (cli.out_weighted()) {
    std::cout << "Weighted graph conversion is not supported yet. Stay tuned...\n";
    assert(false);
    #if 0
    WeightedBuilder bw(cli);
    WGraph wg = bw.MakeGraph();
    wg.PrintStats();
    pvector<NodeID> newIds(wg.num_nodes(), -1);
    WGraph reordered_wg = Builder::degreeSort_weighted(wg, (cli.relabel() == 0), newIds, false, false);
    WeightedWriter ww(reordered_wg);
    std::ofstream output_file(cli.out_filename());
    for (size_t i = 0; i < newIds.size(); i++) {
        output_file << newIds[i] << std::endl;
    }
    output_file.close();
    #endif
  } else {
    Builder b(cli);
    Graph g = b.MakeGraph();
    g.PrintStats();
    pvector<NodeID> newIds(g.num_nodes(), -1);
    Graph reordered_g = Builder::degreeSort(g, (cli.relabel() == 0), newIds, false, true);
    Writer w(reordered_g);
    std::ofstream output_file(cli.out_filename());
    for (size_t i = 0; i < newIds.size(); i++) {
        output_file << newIds[i] << std::endl;
    }
    output_file.close();
  }
  return 0;
}
