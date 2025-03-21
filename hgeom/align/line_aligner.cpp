// format_identifier.cpp
#include <algorithm>
#include <cctype>
#include <iostream>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <ranges>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;
using namespace std;

// Character group indices for substitution matrix
enum CharGroup {
  UPPERCASE = 0,
  LOWERCASE = 1,
  DIGIT = 2,
  WHITESPACE = 3,
  // All Python punctuation characters as separate groups
  PAREN_OPEN = 4,    // (
  PAREN_CLOSE = 5,   // )
  BRACKET_OPEN = 6,  // [
  BRACKET_CLOSE = 7, // ]
  BRACE_OPEN = 8,    // {
  BRACE_CLOSE = 9,   // }
  DOT = 10,          // .
  COMMA = 11,        // ,
  COLON = 12,        // :
  SEMICOLON = 13,    // ;
  PLUS = 14,         // +
  MINUS = 15,        // -
  ASTERISK = 16,     // *
  SLASH = 17,        // /
  BACKSLASH = 18,    //
  VERTICAL_BAR = 19, // |
  AMPERSAND = 20,    // &
  LESS_THAN = 21,    // <
  GREATER_THAN = 22, // >
  EQUAL = 23,        // =
  PERCENT = 24,      // %
  HASH = 25,         // #
  AT_SIGN = 26,      // @
  EXCLAMATION = 27,  // !
  QUESTION = 28,     // ?
  CARET = 29,        // ^
  TILDE = 30,        // ~
  BACKTICK = 31,     // `
  QUOTE_SINGLE = 32, // '
  QUOTE_DOUBLE = 33, // "
  UNDERSCORE = 34,   // _
  DOLLAR = 35,       // $
  OTHER = 36,        // Other characters
  NUM_GROUPS
};

// Get character group for substitution matrix
CharGroup get_char_group(char c) {
  if (isupper(c))
    return UPPERCASE;
  if (islower(c))
    return LOWERCASE;
  if (isdigit(c))
    return DIGIT;
  if (isspace(c))
    return WHITESPACE;

  // Check for specific punctuation
  switch (c) {
  case '(':
    return PAREN_OPEN;
  case ')':
    return PAREN_CLOSE;
  case '[':
    return BRACKET_OPEN;
  case ']':
    return BRACKET_CLOSE;
  case '{':
    return BRACE_OPEN;
  case '}':
    return BRACE_CLOSE;
  case '.':
    return DOT;
  case ',':
    return COMMA;
  case ':':
    return COLON;
  case ';':
    return SEMICOLON;
  case '+':
    return PLUS;
  case '-':
    return MINUS;
  case '*':
    return ASTERISK;
  case '/':
    return SLASH;
  case '\\':
    return BACKSLASH;
  case '|':
    return VERTICAL_BAR;
  case '&':
    return AMPERSAND;
  case '<':
    return LESS_THAN;
  case '>':
    return GREATER_THAN;
  case '=':
    return EQUAL;
  case '%':
    return PERCENT;
  case '#':
    return HASH;
  case '@':
    return AT_SIGN;
  case '!':
    return EXCLAMATION;
  case '?':
    return QUESTION;
  case '^':
    return CARET;
  case '~':
    return TILDE;
  case '`':
    return BACKTICK;
  case '\'':
    return QUOTE_SINGLE;
  case '"':
    return QUOTE_DOUBLE;
  case '_':
    return UNDERSCORE;
  case '$':
    return DOLLAR;
  default:
    return OTHER;
  }
}

// Default substitution matrix (higher score = more similar)
array<array<float, NUM_GROUPS>, NUM_GROUPS> createDefaultSubMatrix() {
  array<array<float, NUM_GROUPS>, NUM_GROUPS> matrix{};

  // Initialize with zeroes
  for (int i = 0; i < NUM_GROUPS; i++) {
    for (int j = 0; j < NUM_GROUPS; j++) {
      matrix[i][j] = 0.0f;
    }
  }

  // Exact matches get 1.0
  for (int i = 0; i < NUM_GROUPS; i++) {
    matrix[i][i] = 1.0f;
  }

  const vector<CharGroup> keyGroups = {
      EQUAL, COLON, COMMA,    BRACKET_OPEN, PAREN_OPEN,
      PLUS,  MINUS, ASTERISK, SLASH,        UPPERCASE};

  for (const auto &group : keyGroups)
    matrix[group][group] = 5.0;
  matrix[EQUAL][EQUAL] = 10.0;

  // Letter case transitions get 0.9
  matrix[UPPERCASE][LOWERCASE] = 0.3f;
  matrix[LOWERCASE][UPPERCASE] = 0.3f;

  // Letters to digits get 0.5
  matrix[UPPERCASE][DIGIT] = 0.2f;
  matrix[LOWERCASE][DIGIT] = 0.2f;
  matrix[DIGIT][UPPERCASE] = 0.2f;
  matrix[DIGIT][LOWERCASE] = 0.2f;

  // Brackets/parentheses/braces are somewhat similar (0.3)
  matrix[PAREN_OPEN][BRACKET_OPEN] = 0.3f;
  matrix[PAREN_OPEN][BRACE_OPEN] = 0.3f;
  matrix[BRACKET_OPEN][PAREN_OPEN] = 0.3f;
  matrix[BRACKET_OPEN][BRACE_OPEN] = 0.3f;
  matrix[BRACE_OPEN][PAREN_OPEN] = 0.3f;
  matrix[BRACE_OPEN][BRACKET_OPEN] = 0.3f;

  matrix[PAREN_CLOSE][BRACKET_CLOSE] = 0.3f;
  matrix[PAREN_CLOSE][BRACE_CLOSE] = 0.3f;
  matrix[BRACKET_CLOSE][PAREN_CLOSE] = 0.3f;
  matrix[BRACKET_CLOSE][BRACE_CLOSE] = 0.3f;
  matrix[BRACE_CLOSE][PAREN_CLOSE] = 0.3f;
  matrix[BRACE_CLOSE][BRACKET_CLOSE] = 0.3f;

  // Operators have some similarity (0.4)
  matrix[PLUS][MINUS] = 0.4f;
  matrix[MINUS][PLUS] = 0.4f;
  matrix[ASTERISK][SLASH] = 0.4f;
  matrix[SLASH][ASTERISK] = 0.4f;
  matrix[LESS_THAN][GREATER_THAN] = 0.4f;
  matrix[GREATER_THAN][LESS_THAN] = 0.4f;

  // Quotes have similarity
  // matrix[QUOTE_SINGLE][QUOTE_DOUBLE] = 0.7f;
  // matrix[QUOTE_DOUBLE][QUOTE_SINGLE] = 0.7f;

  return matrix;
}

/**
 * A more robust implementation that properly handles Python syntax edge cases.
 * Uses state tracking to ensure we only consider real statements, not strings
 * containing colons.
 */
bool isInlineStatementRobust(string const &line) {
  // Skip empty lines
  if (line.empty()) {
    return false;
  }

  // Get a copy of the line without leading whitespace
  string trimmed = line;
  size_t firstNonSpace = trimmed.find_first_not_of(" \t");
  if (firstNonSpace == string::npos) {
    return false; // Empty line
  }

  trimmed = trimmed.substr(firstNonSpace);

  // Check if this is a comment-only line
  if (trimmed[0] == '#') {
    return false;
  }

  // Check for statement keywords at the beginning of the line
  const vector<string> keywords = {
      "if ", "elif ", "else:", "for ", "def ", "class "};

  bool foundKeyword = false;
  string keywordFound;

  for (const auto &keyword : keywords) {
    if (trimmed.compare(0, keyword.length(), keyword) == 0) {
      foundKeyword = true;
      keywordFound = keyword;
      break;
    }
  }

  if (!foundKeyword) {
    return false;
  }

  // Now we need to find the colon that ends the statement header
  size_t colonPos = 0;
  bool inString = false;
  char stringDelimiter = 0;
  bool escaped = false;
  int parenLevel = 0;

  // For else:, we already know the colon position
  if (keywordFound == "else:") {
    colonPos = firstNonSpace + 4; // "else" length
  } else {
    // For other keywords, we need to find the colon
    for (size_t i = 0; i < trimmed.length(); i++) {
      char c = trimmed[i];

      // Handle string delimiters
      if ((c == '"' || c == '\'') && !escaped) {
        if (!inString) {
          inString = true;
          stringDelimiter = c;
        } else if (c == stringDelimiter) {
          inString = false;
        }
      }

      // Handle escaping
      if (c == '\\' && !escaped) {
        escaped = true;
        continue;
      } else {
        escaped = false;
      }

      // Track parentheses level (ignore if in string)
      if (!inString) {
        if (c == '(' || c == '[' || c == '{') {
          parenLevel++;
        } else if (c == ')' || c == ']' || c == '}') {
          parenLevel--;
        } else if (c == ':' && parenLevel == 0) {
          colonPos = firstNonSpace + i;
          break;
        }
      }
    }
  }

  // If we couldn't find a proper colon, it's not a valid statement
  if (colonPos == 0 || colonPos >= line.length() - 1) {
    return false;
  }

  // Now check if there's an action after the colon
  string afterColon = line.substr(colonPos + 1);
  size_t actionStart = afterColon.find_first_not_of(" \t");

  // If there's nothing after the colon or just a comment, it's not an inline
  // action
  if (actionStart == string::npos || afterColon[actionStart] == '#') {
    return false;
  }

  return true;
}

// Structure to keep positions of important formatting characters
struct LineFormatInfo {
  unordered_map<CharGroup, vector<size_t>> charPositions;

  void recordPosition(char c, size_t pos) {
    CharGroup group = get_char_group(c);
    charPositions[group].push_back(pos);
  }
};

// Get formatting information about a line
LineFormatInfo getLineFormatInfo(string const &line) {
  LineFormatInfo info;
  for (size_t i = 0; i < line.size(); i++) {
    info.recordPosition(line[i], i);
  }
  return info;
}

// Get indentation level of a line
string getIndentation(string const &line) {
  auto nonWhitespace = line.find_first_not_of(" \t");
  if (nonWhitespace == string::npos) {
    return "";
  }
  return line.substr(0, nonWhitespace);
}

class IdentifyFormattedBlocks {
public:
  array<array<float, NUM_GROUPS>, NUM_GROUPS> sub_matrix;

  IdentifyFormattedBlocks() { sub_matrix = createDefaultSubMatrix(); }

  void set_substitution_matrix(CharGroup i, CharGroup j, float val) {
    sub_matrix[i][j] = val;
  }

  // Compute similarity score between two lines
  float compute_similarity_score(string const &line1, string const &line2) {
    // Handle empty lines
    if (line1.empty() || line2.empty())
      return 0.0f;

    size_t indent1 = line1.find_first_not_of(" \t");
    size_t indent2 = line2.find_first_not_of(" \t");
    if (indent1 != indent2)
      return 0.0f;

    // Get line format info for special character positions
    LineFormatInfo info1 = getLineFormatInfo(line1);
    LineFormatInfo info2 = getLineFormatInfo(line2);

    // Compute alignment score
    float alignmentScore = 0.0f;

    size_t len1 = line1.size();
    size_t len2 = line2.size();

    // Score character by character for alignment
    for (size_t i = 0; i < min(len1, len2); i++) {
      CharGroup g1 = get_char_group(line1[i]);
      CharGroup g2 = get_char_group(line2[i]);
      alignmentScore += sub_matrix[g1][g2];
    }
    float maxlen = static_cast<float>(max(line1.size(), line2.size()));
    alignmentScore = alignmentScore / sqrt(maxlen);

    // Penalize for length difference
    float lengthPenalty =
        1.0f - (abs(static_cast<int>(len1) - static_cast<int>(len2)) /
                static_cast<float>(max(len1, len2)));

    // Combine scores with weights
    return 0.7f * alignmentScore + 0.3f * lengthPenalty;
  }

  string unmark(string const &code) {
    vector<string> lines;
    istringstream stream(code);
    string line;
    while (getline(stream, line)) {
      lines.push_back(line);
    }
    if (lines.empty()) {
      return code;
    }
    vector<string> output;

    for (string const &line : lines) {
      if (line.find("#             fmt:") == string::npos) {
        output.push_back(line);
      }
    }

    // Join lines back into a string
    ostringstream result;
    for (size_t i = 0; i < output.size(); i++) {
      result << output[i];
      if (i < output.size() - 1) {
        result << "\n";
      }
    }
    return result.str();
  }

  // Process code to identify and mark well-formatted blocks
  string mark_formtted_blocks(string const &code, float threshold = 3.0f) {
    // Parse code into lines
    vector<string> lines;
    istringstream stream(code);
    string line;
    while (getline(stream, line)) {
      lines.push_back(line);
    }
    if (lines.empty()) {
      return code;
    }

    // Output lines
    vector<string> output;

    // Track blocks
    bool inFormattedBlock = false;
    size_t formatBlockStart = 0;
    size_t consecutiveHighScores = 0;
    bool lastwasinline = false;

    // Process each line
    for (size_t i = 1; i < lines.size(); i++) {

      bool isinline = isInlineStatementRobust(lines[i]);
      if (isinline) {
        string indent = getIndentation(lines[i]);
        if (!lastwasinline)
          output.push_back(indent + "#             fmt: off");
        output.push_back(lines[i]);
        lastwasinline = true;
        continue;
      } else if (lastwasinline) {
        string indent = getIndentation(lines[i]);
        lastwasinline = false;
        output.push_back(indent + "#             fmt: on");
      }

      float score = compute_similarity_score(lines[i - 1], lines[i]);

      // Start tracking potential formatted block
      if (score >= threshold) {
        consecutiveHighScores++;

        // If we found 2 or more consecutive high similarity lines
        // and we're not already in a marked block
        if (consecutiveHighScores >= 1 && !inFormattedBlock) {
          inFormattedBlock = true;
          formatBlockStart = i - 1;

          // Get indentation
          string indent = getIndentation(lines[formatBlockStart]);

          // Insert the fmt: off line before the block
          if (output.size())
            output.pop_back();
          output.push_back(indent + "#             fmt: off");
          output.push_back(lines[formatBlockStart]);
        }

        if (inFormattedBlock) {
          output.push_back(lines[i]);
        }
      }
      // If similarity dropped below threshold
      else {
        // If we were in a formatted block, close it
        if (inFormattedBlock) {
          // Get indentation
          string indent = getIndentation(lines.back());

          // Add the fmt: on line
          output.push_back(indent + "#             fmt: on");
          inFormattedBlock = false;
          consecutiveHighScores = 0;
        }
        // Not in a block, just add the lines
        else {
          if (i == 1) {
            output.push_back(lines[0]);
          }
          output.push_back(lines[i]);
          consecutiveHighScores = 0;
        }
      }
    }

    // Close any open block at the end of file
    if (inFormattedBlock) {
      string indent = getIndentation(lines.back());
      output.push_back(indent + "#             fmt: on");
    }

    // Add the first line if it w0asn't part of any block
    if (output.empty() || output[0] != lines[0] &&
                              output[0] != "#             fmt: off" &&
                              output.size() >= 2 && output[1] != lines[0]) {
      output.insert(output.begin(), lines[0]);
    }

    // Join lines back into a string
    ostringstream result;
    for (size_t i = 0; i < output.size(); i++) {
      result << output[i];
      if (i < output.size() - 1) {
        result << "\n";
      }
    }

    return result.str();
  }
};

PYBIND11_MODULE(_line_align, m) {
  m.doc() = "Identifies and marks well-formatted code blocks with fmt: off/on "
            "markers";

  py::class_<IdentifyFormattedBlocks>(m, "IdentifyFormattedBlocks")
      .def(py::init<>(),
           "Default constructor which initializes the substitution matrix.")
      .def("set_substitution_matrix",
           &IdentifyFormattedBlocks::set_substitution_matrix, py::arg("i"),
           py::arg("j"), py::arg("val"),
           "Set a value in the substitution matrix at indices (i, j).")
      .def("compute_similarity_score",
           &IdentifyFormattedBlocks::compute_similarity_score, py::arg("line1"),
           py::arg("line2"), "Compute similarity score between two lines")
      .def("mark_formtted_blocks",
           &IdentifyFormattedBlocks::mark_formtted_blocks, py::arg("code"),
           py::arg("threshold") = 0.7f,
           "Process the input code and mark formatted blocks based on a "
           "similarity threshold.")
      .def("unmark", &IdentifyFormattedBlocks::unmark, py::arg("code"),
           "remove marks.");

  py::enum_<CharGroup>(m, "CharGroup")
      .value("UPPERCASE", UPPERCASE)
      .value("LOWERCASE", LOWERCASE)
      .value("DIGIT", DIGIT)
      .value("WHITESPACE", WHITESPACE)
      .value("PAREN_OPEN", PAREN_OPEN)
      .value("PAREN_CLOSE", PAREN_CLOSE)
      .value("BRACKET_OPEN", BRACKET_OPEN)
      .value("BRACKET_CLOSE", BRACKET_CLOSE)
      .value("BRACE_OPEN", BRACE_OPEN)
      .value("BRACE_CLOSE", BRACE_CLOSE)
      .value("DOT", DOT)
      .value("COMMA", COMMA)
      .value("COLON", COLON)
      .value("SEMICOLON", SEMICOLON)
      .value("PLUS", PLUS)
      .value("MINUS", MINUS)
      .value("ASTERISK", ASTERISK)
      .value("SLASH", SLASH)
      .value("BACKSLASH", BACKSLASH)
      .value("VERTICAL_BAR", VERTICAL_BAR)
      .value("AMPERSAND", AMPERSAND)
      .value("LESS_THAN", LESS_THAN)
      .value("GREATER_THAN", GREATER_THAN)
      .value("EQUAL", EQUAL)
      .value("PERCENT", PERCENT)
      .value("HASH", HASH)
      .value("AT_SIGN", AT_SIGN)
      .value("EXCLAMATION", EXCLAMATION)
      .value("QUESTION", QUESTION)
      .value("CARET", CARET)
      .value("TILDE", TILDE)
      .value("BACKTICK", BACKTICK)
      .value("QUOTE_SINGLE", QUOTE_SINGLE)
      .value("QUOTE_DOUBLE", QUOTE_DOUBLE)
      .value("UNDERSCORE", UNDERSCORE)
      .value("DOLLAR", DOLLAR)
      .value("OTHER", OTHER)
      .value("NUM_GROUPS", NUM_GROUPS)
      .export_values(); // Export values to the module scope
}
