// Minimal readers for the flat JSON files this app loads (direct.json, rig.json): the first value
// of a key anywhere in the text. Enough for machine-written files; not a general JSON parser.
#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace hand_master::json {

inline std::size_t value_start(const std::string& text, const std::string& key) {
    const std::string quoted = "\"" + key + "\"";
    const auto at = text.find(quoted);
    if (at == std::string::npos) throw std::runtime_error("Missing JSON key " + key);
    const auto colon = text.find(':', at + quoted.size());
    if (colon == std::string::npos) throw std::runtime_error("Malformed JSON at key " + key);
    return text.find_first_not_of(" \t\r\n", colon + 1);
}

inline double number(const std::string& text, const std::string& key) {
    const auto start = value_start(text, key);
    const char* begin = text.c_str() + start;
    char* end = nullptr;
    const double value = std::strtod(begin, &end);
    if (end == begin) throw std::runtime_error("JSON key " + key + " is not a number");
    return value;
}

inline std::string string(const std::string& text, const std::string& key) {
    const auto start = value_start(text, key);
    if (text[start] != '"') throw std::runtime_error("JSON key " + key + " is not a string");
    const auto end = text.find('"', start + 1);
    return text.substr(start + 1, end - start - 1);
}

inline bool boolean(const std::string& text, const std::string& key) {
    const auto start = value_start(text, key);
    if (text.compare(start, 4, "true") == 0) return true;
    if (text.compare(start, 5, "false") == 0) return false;
    throw std::runtime_error("JSON key " + key + " is not true/false");
}

// Every number inside the (possibly nested) array value of key, in order.
inline std::vector<double> numbers(const std::string& text, const std::string& key) {
    auto at = value_start(text, key);
    if (text[at] != '[') throw std::runtime_error("JSON key " + key + " is not an array");
    std::vector<double> values;
    int depth = 0;
    for (; at < text.size(); ++at) {
        const char c = text[at];
        if (c == '[') ++depth;
        else if (c == ']') { if (--depth == 0) break; }
        else if (c == '-' || c == '+' || c == '.' || (c >= '0' && c <= '9')) {
            char* end = nullptr;
            values.push_back(std::strtod(text.c_str() + at, &end));
            at = static_cast<std::size_t>(end - text.c_str()) - 1;
        }
    }
    if (depth != 0) throw std::runtime_error("Unterminated JSON array at key " + key);
    return values;
}

}  // namespace hand_master::json
