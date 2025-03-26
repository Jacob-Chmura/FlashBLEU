#include <fmt/core.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string_view>
#include <unordered_map>
#include <vector>

class ScopedTimer {
 public:
  using ClockType = std::chrono::steady_clock;

  explicit ScopedTimer(const char* func_name)
      : func_name_(func_name), st_{ClockType::now()} {}

  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer(ScopedTimer&&) = delete;
  auto operator=(const ScopedTimer&) -> ScopedTimer& = delete;
  auto operator=(const ScopedTimer&&) -> ScopedTimer& = delete;

  ~ScopedTimer() {
    std::chrono::duration<double> s = ClockType::now() - st_;
    std::cout << fmt::format("{}: {} s", func_name_, s.count()) << "\n";
  }

 private:
  const std::string func_name_{};
  const ClockType::time_point st_{};
};

namespace {
auto tokenize(const std::string& input, char delim = ' ')
    -> std::vector<std::string_view> {
  std::vector<std::string_view> out;
  for (const auto& token : input | std::views::split(delim)) {
    out.emplace_back(&*token.begin(), std::ranges::distance(token));
  }
  return out;
}

auto count_ngram(const std::vector<std::string_view>& tokens, size_t n_gram)
    -> std::unordered_map<std::string, size_t> {
  std::unordered_map<std::string, size_t> count;
  for (size_t i = 1; i < n_gram + 1; ++i) {
    for (size_t j = 0; j < tokens.size() - i + 1; ++j) {
      // Hash the string: TODO
      std::string ngram_string;
      for (size_t k = j; k < i + j; ++k) {
        ngram_string += tokens[k];
        ngram_string += "|";  // assumes no pipes
      }
      count[ngram_string] += 1;
    }
  }
  return count;
}

auto bleu_score(const std::vector<std::string>& preds,
                const std::vector<std::vector<std::string>>& targets,
                size_t n_gram = 4,
                bool smooth = false) {
  const std::vector<float> weights(n_gram, 1.0F / static_cast<float>(n_gram));

  auto st = std::chrono::high_resolution_clock::now();

  std::vector<std::vector<std::string_view>> tokenized_preds(preds.size());
  for (size_t i = 0; i < preds.size(); ++i) {
    tokenized_preds[i] = tokenize(preds[i]);
  }
  std::vector<std::vector<std::vector<std::string_view>>> token_targets(
      targets.size());
  for (size_t i = 0; i < targets.size(); ++i) {
    token_targets[i] =
        std::vector<std::vector<std::string_view>>(targets[i].size());
    for (size_t j = 0; j < targets[i].size(); ++j) {
      token_targets[i][j] = tokenize(targets[i][j]);
    }
  }

  std::chrono::duration<double> ms =
      std::chrono::high_resolution_clock::now() - st;
  std::cout << fmt::format("Tokenization: {} s", ms.count()) << "\n";

  st = std::chrono::high_resolution_clock::now();

  std::vector<size_t> num(n_gram);
  std::vector<size_t> denom(n_gram);

  size_t preds_len{0};
  size_t targets_len{0};
  std::vector<size_t> len_diffs(token_targets.size());
  for (size_t i = 0; i < preds.size(); ++i) {
    len_diffs.clear();
    const size_t pred_len = tokenized_preds[i].size();
    preds_len += pred_len;
    std::transform(
        token_targets[i].begin(), token_targets[i].end(), len_diffs.begin(),
        [pred_len](const auto& x) {
          return abs(static_cast<int>(x.size()) - static_cast<int>(pred_len));
        });
    auto min_idx =
        std::distance(len_diffs.begin(), std::ranges::min_element(len_diffs));
    targets_len += len_diffs[min_idx];

    auto pred_n_gram = count_ngram(tokenized_preds[i], n_gram);

    std::unordered_map<std::string, size_t> targets_n_gram;
    for (const auto& tokenized_target : token_targets[i]) {
      auto target_n_gram = count_ngram(tokenized_target, n_gram);
      for (const auto& [k, v] : target_n_gram) {
        if (targets_n_gram.contains(k)) {
          targets_n_gram[k] = std::max(targets_n_gram[k], v);
        } else {
          targets_n_gram[k] = v;
        }
      }
    }

    std::unordered_map<std::string, size_t> n_gram;
    for (const auto& [k, v] : targets_n_gram) {
      if (pred_n_gram.contains(k)) {
        n_gram[k] = std::min(pred_n_gram[k], v);
      }
    }
    for (const auto& [k, v] : pred_n_gram) {
      if (targets_n_gram.contains(k)) {
        n_gram[k] = std::min(targets_n_gram[k], v);
      }
    }

    for (const auto& [k, v] : n_gram) {
      num[std::ranges::count(k, '|') - 1] += v;
    }
    for (const auto& [k, v] : pred_n_gram) {
      denom[std::ranges::count(k, '|') - 1] += v;
    }
  }

  ms = std::chrono::high_resolution_clock::now() - st;
  std::cout << fmt::format("Ngrams: {} s", ms.count()) << "\n";

  st = std::chrono::high_resolution_clock::now();

  std::vector<double> precision(n_gram);
  if (smooth) {
    std::ranges::transform(num, denom, precision.begin(),
                           [](const auto& n_val, const auto& denom_val) {
                             return static_cast<double>((n_val + 1)) /
                                    (denom_val + 1);
                           });
    precision[0] = static_cast<double>(num[0]) / static_cast<double>(denom[0]);
  } else {
    std::ranges::transform(num, denom, precision.begin(),
                           [](const auto& n_val, const auto& denom_val) {
                             return static_cast<double>(n_val) / denom_val;
                           });
  }

  std::ranges::transform(precision, weights, precision.begin(),
                         [](auto& precision_val, auto& weight_val) {
                           return weight_val * std::log(precision_val);
                         });

  const double geometric_mean =
      exp(std::reduce(precision.begin(), precision.end()));
  const double brevity_penalty =
      preds_len > targets_len ? 1
                              : exp(1 - (static_cast<double>(targets_len) /
                                         static_cast<double>(preds_len)));
  const double bleu = brevity_penalty * geometric_mean;

  ms = std::chrono::high_resolution_clock::now() - st;
  std::cout << fmt::format("Bleu: {} s", ms.count()) << "\n";
  return bleu;
}

auto run_torchmetrics_example() -> void {
  const std::vector<std::string> preds = {"the squirrel is eating the nut",
                                          "the cat is on the mat"};
  const std::vector<std::vector<std::string>> references = {
      {"a squirrel is eating a nut", "the squirrel is eating a tasty nut"},
      {"there is a cat on the mat", "a cat is on the mat"}};

  std::cout << bleu_score(preds, references) << "\n";
}

auto run_synthetic_example() -> void {
  const auto n_tokens{1000};
  const auto n_samples{100};

  const std::string token{"foo "};
  std::string s;
  s.reserve(token.size() * n_tokens);
  for (auto i = 0; i < n_tokens; ++i) {
    s += token;
  }

  const std::vector<std::string> preds(n_samples, s);
  const std::vector<std::vector<std::string>> targets(
      n_samples, std::vector<std::string>(n_samples, s));

  bleu_score(preds, targets);
}
}  // namespace

auto main() -> int {
  run_torchmetrics_example();
  run_synthetic_example();
}
