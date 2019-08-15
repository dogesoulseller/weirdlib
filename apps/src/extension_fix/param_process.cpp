#include "param_process.hpp"

bool StringIsInteger(const std::string& str) {
	if (str[0] == '-' || std::isdigit(str[0]) == 0) {
		return false;
	} else {
		return std::all_of(std::next(str.cbegin()), str.cend(), [](const char c) -> bool {return std::isdigit(c) != 0;});
	}
}

std::unordered_map<ParameterName, std::string> GetParameters(std::vector<std::string>& args) {
	std::unordered_map<ParameterName, std::string> Parameters;
	Parameters.insert(std::pair(RECURSION_DEPTH, "-1"));
	Parameters.insert(std::pair(OUTPUT_DIR, "efix"));

	for (size_t i = 2; i < args.size(); i++) {
		if (args[i] == "-d" || args[i] == "--depth") {
			if (i == args.size()-1) {
				std::cerr << "No recursion depth passed\n";
				std::exit(2);
			}

			i++;
			if (!StringIsInteger(args[i])) {
				std::cerr << "Value " << args[i] << " was not recognized as a valid recursion depth";
				std::exit(2);
			}

			Parameters.insert_or_assign(RECURSION_DEPTH, args[i]);
		} else if (args[i] == "-o" || args[i] == "--output") {
			if (i == args.size()-1) {
				std::cerr << "No output directory passed\n";
				std::exit(2);
			}

			i++;

			Parameters.insert_or_assign(OUTPUT_DIR, args[i]);
		} else {
			std::cerr << "Unknown switch " << args[i] << '\n';
			std::exit(2);
		}
	}

	return Parameters;
}
