#ifndef CSVREADER_H
#define CSVREADER_H
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <string>
#include <map>
#include <algorithm>

using std::string;
using std::ifstream;
using std::fstream;
using std::ostringstream;
using std::istringstream;
using std::vector;
using std::map;

inline string readFileIntoString(const string& path) {
    auto ss = ostringstream{};
    ifstream input_file(path);
    if (!input_file.is_open()) {
        throw std::runtime_error("Could not open the file - '" + path);
    }
    ss << input_file.rdbuf();
    return ss.str();
}

namespace util {

inline double read_record(std::string filename, int rownum, int colnum) {
    string file_contents, line;
    map<int, vector<string>> csv_contents;
    char delimiter = ',';

    file_contents = readFileIntoString(filename);

    istringstream sstream(file_contents);
    std::vector<string> items;
    string record;
    int counter = 0;
    while (std::getline(sstream, record)) {
        istringstream line(record);
        while (std::getline(line, record, delimiter)) {
            record.erase(std::remove_if(record.begin(), record.end(), isspace), record.end());
            items.push_back(record);
        }
        csv_contents[counter] = items;
        items.clear();
        counter += 1;
    }
    auto numstr = csv_contents[rownum][colnum];
    double acc = stod(numstr);
    return acc;
}

}//namespace util 

#endif//CSVREADER_H
