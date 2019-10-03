from sys import version_info, exit, argv
if version_info < (3, 5):
	exit("This script requires Python 3.5 or newer!")

from os import path, walk
from subprocess import run
from time import time_ns

accepted_extensions = (".c", ".cc", ".cxx", ".cpp", ".C", ".h", ".hh", ".hpp", ".hxx", ".H")
directories_to_check = ["apps", "include", "src", "benchmark"]

checks = "bugprone-*,"\
	+"modernize-*,-modernize-use-trailing-return-type,-modernize-concat-nested-namespaces,"\
	+"performance-*,"\
	+"readability-*,-readability-magic-numbers,-readability-uppercase-literal-suffix,"\
	+"portability-*,-portability-simd-intrinsics,"\
	+"clang-analyzer-*,"\
	+"misc-*,"\
	+"-clang-analyzer-osx*,-clang-analyzer-optin.osx*,-clang-analyzer-optin.mpi*,-clang-analyzer-apiModeling*"

# Get dir where path is
def get_source_dir():
	return path.abspath(path.dirname(argv[0]))

# Get all source and header files
def get_all_source_files():
	fs = []
	for directory in directories_to_check:
		for (dirpath, _, fnames) in walk(path.join(get_source_dir(), directory)):
			for fname in fnames:
				if path.join(dirpath,fname).endswith(accepted_extensions):
					fs.append(path.join(dirpath, fname))
	return fs

# If script had any options passed, interpret them as a list of files to process
files = []
if len(argv) == 1:
	files = get_all_source_files()
else:
	files.extend(map(path.abspath, argv[1:]))

for file_name in files:
	print("\nNow processing file: " + file_name, flush=True)
	print("--------------------------------------------------------------------------------")

	start_time = time_ns()
	run(["clang-tidy", file_name, "-p", get_source_dir(), "--checks={}".format(checks),
		"--", "-march=native", "-std=c++17", "-Iinclude"])
	end_time = time_ns()

	print("Finished processing file in {:.3f} ms".format((end_time-start_time)/1000000))
	print("\n--------------------------------------------------------------------------------\n\n")

	try:
		input("Press enter/return to process next file...")
	except EOFError:
		pass
	except SyntaxError:
		pass
