#pragma once

inline static constexpr const char* helpMessage = "Usage: extension-fix FILE/DIR [options]\n"
"FILE - Process single file and exit\n"
"DIR - Recursively process all files in directory\n"
"Options: [-1 denotes infinity where supported]\n"
"    -d | --depth - maximum depth of directory recursion (0 is only base directory) [default: -1]\n"
"    -o | --output - directory for output (either relative or absolute) [default: efix]\n"
"       | --append-string - string to append to file name if a file of the same name already exists [default: _other]\n"
"       | --prepend-string - string to prepend to file name if a file of the same name already exists [default: ]\n"
;
