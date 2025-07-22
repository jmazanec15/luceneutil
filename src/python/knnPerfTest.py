#!/usr/bin/env/python

import multiprocessing
import re
import subprocess
import sys
import argparse

import benchUtil
import constants
from common import getLuceneDirFromGradleProperties

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run KNN benchmark')
parser.add_argument('jvm_size', help='JVM heap size (e.g., 8g)')
parser.add_argument('--bp', choices=['true', 'false'], default='true', help='Use best point')
args = parser.parse_args()

DO_PROFILING = False
NOISY = True

# test parameters
PARAMS = {
  'ndoc': (10000000,),
  "maxConn": (16,),
  "beamWidthIndex": (256,),
  "fanout": (256,),
  "numMergeWorker": (12,),
  "numMergeThread": (12,),
  "numSearchThread": (0,2,4,-1),
  "encoding": ("float32",),
  'metric': ('mip',),
  "topK": (100,),
  "bp": (args.bp,),
  "quantizeCompress": (True,),
  "queryStartIndex": (0,),
  "quantizeBits": (1,),
  "forceMerge": (True,),
  'niter': (10000,),
}

OUTPUT_HEADERS = [
  "recall",
  "latency(ms)",
  "netCPU",
  "avgCpuCount",
  "nDoc",
  "topK",
  "fanout",
  "maxConn",
  "beamWidth",
  "quantized",
  "visited",
  "index(s)",
  "index_docs/s",
  "force_merge(s)",
  "num_segments",
  "index_size(MB)",
  "selectivity",
  "filterType",
  "overSample",
  "vec_disk(MB)",
  "vec_RAM(MB)",
  "indexType",
]

def advance(ix, values):
  for i in reversed(range(len(ix))):
    param = list(values.keys())[i]
    if type(values[param]) in (list, tuple) and ix[i] == len(values[param]) - 1:
      ix[i] = 0
    else:
      ix[i] += 1
      return True
  return False

def run_knn_benchmark(checkout, values):
  indexes = [0] * len(values.keys())
  indexes[-1] = -1

  dim = 768
  doc_vectors = "/data/cohere-10m_train.vec"
  query_vectors = "/data/cohere-10m_test.vec"

  jfr_output = f"{constants.LOGS_DIR}/knn-perf-test.jfr"

  cp = benchUtil.classPathToString(benchUtil.getClassPath(checkout) + (f"{constants.BENCH_BASE_DIR}/build",))
  cmd = constants.JAVA_EXE.split(" ") + [
    f"-Xmx{args.jvm_size}",
    f"-Xms{args.jvm_size}",
    "-XX:+AlwaysPreTouch",
    "-cp",
    cp,
    "--add-modules",
    "jdk.incubator.vector",
    "--enable-native-access=ALL-UNNAMED",
    f"-Djava.util.concurrent.ForkJoinPool.common.parallelism={multiprocessing.cpu_count()}",
    "-XX:+UnlockDiagnosticVMOptions",
    "-XX:+DebugNonSafepoints",
  ]

  if DO_PROFILING:
    cmd += [f"-XX:StartFlightRecording=dumponexit=true,maxsize=250M,settings={constants.BENCH_BASE_DIR}/src/python/profiling.jfc" + f",filename={jfr_output}"]

  cmd += ["knn.KnnGraphTester"]

  all_results = []
  while advance(indexes, values):
    if NOISY:
      print("\nNEXT:")
    pv = {}
    cmd_args = []
    quantize_bits = None
    do_quantize_compress = False
    for i, p in enumerate(values.keys()):
      if values[p]:
        value = values[p][indexes[i]]
        if p == "quantizeBits":
          if value != 32:
            pv[p] = value
            print(f"  -{p}={value}")
            print("  -quantize")
            cmd_args += ["-quantize"]
            quantize_bits = value
        elif type(value) is bool:
          if p == "quantizeCompress":
            do_quantize_compress = True
          elif value:
            cmd_args += ["-" + p]
            print(f"  -{p}")
        else:
          print(f"  -{p}={value}")
          pv[p] = value
      else:
        cmd_args += ["-" + p]
        print(f"  -{p}")

    if quantize_bits == 4 and do_quantize_compress:
      cmd_args += ["-quantizeCompress"]
      print("  -quantizeCompress")

    cmd_args += [a for (k, v) in pv.items() for a in ("-" + k, str(v)) if a]

    # Add basic arguments
    mode_args = [
      "-dim",
      str(dim),
      "-docs",
      doc_vectors,
      "-numIndexThreads",
      "8",
      "-reindex",  # Always include reindex
      "-search-and-stats",  # Always include search
      query_vectors
    ]

    this_cmd = cmd + cmd_args + mode_args

    if NOISY:
      print(f"  cmd: {this_cmd}")
    else:
      cmd += ["-quiet"]

    job = subprocess.Popen(this_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8")
    re_summary = re.compile(r"^SUMMARY: (.*?)$", re.MULTILINE)
    summary = None
    lines = ""
    while True:
      line = job.stdout.readline()
      if line == "":
        break
      lines += line
      if NOISY:
        sys.stdout.write(line)
      m = re_summary.match(line)
      if m is not None:
        summary = m.group(1)

    job.wait()
    if job.returncode != 0:
      print(f"Command failed with exit {job.returncode}")
      print(f"Output: {lines}")
      continue

    if summary is None:
      print("No summary found in output")
      continue

    all_results.append((summary, cmd_args))
    if DO_PROFILING:
      benchUtil.profilerOutput(constants.JAVA_EXE, jfr_output, benchUtil.checkoutToPath(checkout), 30, (1,))

  if NOISY:
    print("\nResults:")

  skip_headers = {"selectivity", "filterType", "visited"}

  if "-forceMerge" not in this_cmd:
    skip_headers.add("force_merge(s)")
  if "-overSample" not in this_cmd:
    skip_headers.add("overSample")
  if "-indexType" in this_cmd and "flat" in this_cmd:
    skip_headers.add("maxConn")
    skip_headers.add("beamWidth")

  print_fixed_width(all_results, skip_headers)
  print_chart(all_results)

def print_fixed_width(all_results, columns_to_skip):
  header = "\t".join(
    h for h in OUTPUT_HEADERS if h not in columns_to_skip
  )
  print(header)
  for summary, args in all_results:
    print(summary)

def print_chart(all_results):
  pass

if __name__ == "__main__":
  LUCENE_CHECKOUT = getLuceneDirFromGradleProperties()
  run_knn_benchmark(LUCENE_CHECKOUT, PARAMS)
