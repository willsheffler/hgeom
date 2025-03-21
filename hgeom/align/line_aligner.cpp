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
CharGroup getCharGroup(char c) {
  if (std::isupper(c))
    return UPPERCASE;
  if (std::islower(c))
    return LOWERCASE;
  if (std::isdigit(c))
    return DIGIT;
  if (std::isspace(c))
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
std::array<std::array<float, NUM_GROUPS>, NUM_GROUPS> createDefaultSubMatrix() {
  std::array<std::array<float, NUM_GROUPS>, NUM_GROUPS> matrix{};

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

  // Letter case transitions get 0.9
  matrix[UPPERCASE][LOWERCASE] = 0.9f;
  matrix[LOWERCASE][UPPERCASE] = 0.9f;

  // Letters to digits get 0.5
  matrix[UPPERCASE][DIGIT] = 0.5f;
  matrix[LOWERCASE][DIGIT] = 0.5f;
  matrix[DIGIT][UPPERCASE] = 0.5f;
  matrix[DIGIT][LOWERCASE] = 0.5f;

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
  matrix[QUOTE_SINGLE][QUOTE_DOUBLE] = 0.7f;
  matrix[QUOTE_DOUBLE][QUOTE_SINGLE] = 0.7f;

  return matrix;
}

/**
 * A more robust implementation that properly handles Python syntax edge cases.
 * Uses state tracking to ensure we only consider real statements, not strings
 * containing colons.
 */
bool isInlineStatementRobust(const std::string &line) {
  // Skip empty lines
  if (line.empty()) {
    return false;
  }

  // Get a copy of the line without leading whitespace
  std::string trimmed = line;
  size_t firstNonSpace = trimmed.find_first_not_of(" \t");
  if (firstNonSpace == std::string::npos) {
    return false; // Empty line
  }

  trimmed = trimmed.substr(firstNonSpace);

  // Check if this is a comment-only line
  if (trimmed[0] == '#') {
    return false;
  }

  // Check for statement keywords at the beginning of the line
  const std::vector<std::string> keywords = {
      "if ", "elif ", "else:", "for ", "def ", "class "};

  bool foundKeyword = false;
  std::string keywordFound;

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
  std::string afterColon = line.substr(colonPos + 1);
  size_t actionStart = afterColon.find_first_not_of(" \t");

  // If there's nothing after the colon or just a comment, it's not an inline
  // action
  if (actionStart == std::string::npos || afterColon[actionStart] == '#') {
    return false;
  }

  return true;
}

// Structure to keep positions of important formatting characters
struct LineFormatInfo {
  std::unordered_map<CharGroup, std::vector<size_t>> charPositions;

  void recordPosition(char c, size_t pos) {
    CharGroup group = getCharGroup(c);
    charPositions[group].push_back(pos);
  }
};

// Get formatting information about a line
LineFormatInfo getLineFormatInfo(const std::string &line) {
  LineFormatInfo info;
  for (size_t i = 0; i < line.size(); i++) {
    info.recordPosition(line[i], i);
  }
  return info;
}

// Compute similarity score between two lines
float computeSimilarityScore(
    const std::string &line1, const std::string &line2,
    const std::array<std::array<float, NUM_GROUPS>, NUM_GROUPS> &subMatrix) {
  // Handle empty lines
  if (line1.empty() || line2.empty())
    return 0.0f;

  // Get line format info for special character positions
  LineFormatInfo info1 = getLineFormatInfo(line1);
  LineFormatInfo info2 = getLineFormatInfo(line2);

  // Compute alignment score
  float alignmentScore = 0.0f;
  float maxPossibleScore =
      static_cast<float>(std::max(line1.size(), line2.size()));

  size_t len1 = line1.size();
  size_t len2 = line2.size();
  size_t minLen = std::min(len1, len2);

  // Score character by character for alignment
  for (size_t i = 0; i < minLen; i++) {
    CharGroup g1 = getCharGroup(line1[i]);
    CharGroup g2 = getCharGroup(line2[i]);
    alignmentScore += subMatrix[g1][g2];
  }

  // Penalize for length difference
  float lengthPenalty =
      1.0f - (std::abs(static_cast<int>(len1) - static_cast<int>(len2)) /
              static_cast<float>(std::max(len1, len2)));

  // Pattern recognition for alignment
  float patternScore = 0.0f;
  float patternWeight = 0.0f;

  // Check for aligned key characters
  // We give special attention to characters that are often aligned in formatted
  // code
  const std::vector<CharGroup> keyGroups = {EQUAL,        COLON,      COMMA,
                                            BRACKET_OPEN, PAREN_OPEN, PLUS,
                                            MINUS,        ASTERISK,   SLASH};

  for (const auto &group : keyGroups) {
    const auto &positions1 = info1.charPositions[group];
    const auto &positions2 = info2.charPositions[group];

    // If both lines have this character
    if (!positions1.empty() && !positions2.empty()) {
      float bestMatch = 0.0f;

      // Find the best matching positions
      for (auto pos1 : positions1) {
        for (auto pos2 : positions2) {
          float posSimilarity =
              1.0f -
              std::abs(static_cast<float>(pos1) - static_cast<float>(pos2)) /
                  static_cast<float>(std::max(len1, len2));

          // Give more weight to certain characters
          float charWeight = 1.0f;
          if (group == EQUAL)
            charWeight = 3.0f; // '=' is often aligned
          else if (group == COLON)
            charWeight = 2.5f; // ':' is often aligned
          else if (group == COMMA)
            charWeight = 1.5f;

          float matchScore = posSimilarity * charWeight;
          bestMatch = std::max(bestMatch, matchScore);
        }
      }

      patternScore += bestMatch;
      patternWeight += 1.0f;
    }
  }

  // Normalize pattern score if we found patterns
  float normalizedPatternScore =
      patternWeight > 0 ? patternScore / patternWeight : 0.0f;

  // Check for similar indentation
  size_t indent1 = line1.find_first_not_of(" \t");
  size_t indent2 = line2.find_first_not_of(" \t");

  float indentationScore = 0.0f;
  if (indent1 != std::string::npos && indent2 != std::string::npos) {
    indentationScore = 1.0f - std::abs(static_cast<float>(indent1) -
                                       static_cast<float>(indent2)) /
                                  static_cast<float>(std::max(len1, len2));
  }

  // Normalize alignment score
  alignmentScore = alignmentScore / maxPossibleScore;

  // Combine scores with weights
  return 0.3f * alignmentScore + 0.4f * normalizedPatternScore +
         0.2f * lengthPenalty + 0.1f * indentationScore;
}

// Get indentation level of a line
std::string getIndentation(const std::string &line) {
  auto nonWhitespace = line.find_first_not_of(" \t");
  if (nonWhitespace == std::string::npos) {
    return "";
  }
  return line.substr(0, nonWhitespace);
}

// Process code to identify and mark well-formatted blocks
std::string identifyFormattedBlocks(const std::string &code,
                                    float threshold = 0.7f) {
  // Parse code into lines
  std::vector<std::string> lines;
  std::istringstream stream(code);
  std::string line;

  while (std::getline(stream, line)) {
    lines.push_back(line);
  }

  // Empty input case
  if (lines.empty()) {
    return code;
  }

  // Default substitution matrix
  auto subMatrix = createDefaultSubMatrix();

  // Output lines
  std::vector<std::string> outputLines;

  // Track blocks
  bool inFormattedBlock = false;
  size_t formatBlockStart = 0;
  size_t consecutiveHighScores = 0;
  bool lastwasinline = false;

  // Process each line
  for (size_t i = 1; i < lines.size(); i++) {

    bool isinline = isInlineStatementRobust(lines[i]);
    if (isinline) {
      std::string indent = getIndentation(lines[i]);
      if (!lastwasinline)
        outputLines.push_back(indent + "#             fmt: off");
      outputLines.push_back(lines[i]);
      lastwasinline = true;
      continue;
    } else if (lastwasinline) {
      std::string indent = getIndentation(lines[i]);
      lastwasinline = false;
      outputLines.push_back(indent + "#             fmt: on");
    }

    float score = computeSimilarityScore(lines[i - 1], lines[i], subMatrix);

    // Start tracking potential formatted block
    if (score >= threshold) {
      consecutiveHighScores++;

      // If we found 2 or more consecutive high similarity lines
      // and we're not already in a marked block
      if (consecutiveHighScores >= 1 && !inFormattedBlock) {
        inFormattedBlock = true;
        formatBlockStart = i - 1;

        // Get indentation
        std::string indent = getIndentation(lines[formatBlockStart]);

        // Insert the fmt: off line before the block
        outputLines.pop_back();
        outputLines.push_back(indent + "#             fmt: off");
        outputLines.push_back(lines[formatBlockStart]);
      }

      if (inFormattedBlock) {
        outputLines.push_back(lines[i]);
      }
    }
    // If similarity dropped below threshold
    else {
      // If we were in a formatted block, close it
      if (inFormattedBlock) {
        // Get indentation
        std::string indent = getIndentation(lines.back());

        // Add the fmt: on line
        outputLines.push_back(indent + "#             fmt: on");
        inFormattedBlock = false;
        consecutiveHighScores = 0;
      }
      // Not in a block, just add the lines
      else {
        if (i == 1) {
          outputLines.push_back(lines[0]);
        }
        outputLines.push_back(lines[i]);
        consecutiveHighScores = 0;
      }
    }
  }

  // Close any open block at the end of file
  if (inFormattedBlock) {
    std::string indent = getIndentation(lines.back());
    outputLines.push_back(indent + "#             fmt: on");
  }

  // Add the first line if it wasn't part of any block
  if (outputLines.empty() || outputLines[0] != lines[0] &&
                                 outputLines[0] != "#             fmt: off" &&
                                 outputLines.size() >= 2 &&
                                 outputLines[1] != lines[0]) {
    outputLines.insert(outputLines.begin(), lines[0]);
  }

  // Join lines back into a string
  std::ostringstream result;
  for (size_t i = 0; i < outputLines.size(); i++) {
    result << outputLines[i];
    if (i < outputLines.size() - 1) {
      result << "\n";
    }
  }

  return result.str();
}

PYBIND11_MODULE(_line_align, m) {
  m.doc() = "Identifies and marks well-formatted code blocks with fmt: off/on "
            "markers";

  m.def("identify_formatted_blocks", &identifyFormattedBlocks,
        "Identifies and marks well-formatted code blocks", py::arg("code"),
        py::arg("threshold") = 0.7f);
}
