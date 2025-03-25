#include <fmt/core.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string_view>
#include <unordered_map>
#include <vector>

auto tokenize(const std::string& input) -> std::vector<std::string_view> {
  std::vector<std::string_view> out;
  size_t start = 0;
  size_t end = input.find(' ');
  while (end != std::string::npos) {
    out.emplace_back(input.data() + start, end - start);
    start = end + 1;
    end = input.find(' ', start);
  }
  out.emplace_back(input.data() + start, input.length() - start);
  return out;
}

auto count_ngram(const std::vector<std::string_view>& tokens, size_t n_gram)
    -> std::unordered_map<std::string, size_t> {
  std::unordered_map<std::string, size_t> count;
  for (size_t i = 1; i < n_gram + 1; ++i) {
    for (size_t j = 0; j < tokens.size() - i + 1; ++j) {
      // Hash the string: TODO
      std::string s;
      for (size_t k = j; k < i + j; ++k) {
        s += tokens[k];
        s += "|";  // assumes no pipes
      }
      count[s] += 1;
    }
  }
  return count;
}

auto bleu_score(const std::vector<std::string>& preds,
                const std::vector<std::vector<std::string>>& targets,
                size_t n_gram = 4,
                bool smooth = false) {
  const std::vector<float> weights(n_gram, 1.0 / n_gram);

  auto st = std::chrono::high_resolution_clock::now();

  std::vector<std::vector<std::string_view>> tokenized_preds(preds.size());
  for (size_t i = 0; i < preds.size(); ++i) {
    tokenized_preds[i] = tokenize(preds[i]);
  }
  std::vector<std::vector<std::vector<std::string_view>>> tokenized_targets(
      targets.size());
  for (size_t i = 0; i < targets.size(); ++i) {
    tokenized_targets[i] =
        std::vector<std::vector<std::string_view>>(targets[i].size());
    for (size_t j = 0; j < targets[i].size(); ++j) {
      tokenized_targets[i][j] = tokenize(targets[i][j]);
    }
  }

  auto et = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> ms = et - st;
  std::cout << fmt::format("Tokenization: {} s", ms.count()) << std::endl;

  st = std::chrono::high_resolution_clock::now();

  std::vector<size_t> num(n_gram);
  std::vector<size_t> denom(n_gram);

  size_t preds_len{0};
  size_t targets_len{0};
  std::vector<size_t> len_diffs(tokenized_targets.size());
  for (size_t i = 0; i < preds.size(); ++i) {
    len_diffs.clear();
    size_t pred_len = tokenized_preds[i].size();
    preds_len += pred_len;
    std::transform(tokenized_targets[i].begin(), tokenized_targets[i].end(),
                   len_diffs.begin(),
                   [pred_len](auto x) { return abs(x.size() - pred_len); });
    auto min_iter = std::min_element(len_diffs.begin(), len_diffs.end());
    size_t min_idx = std::distance(len_diffs.begin(), min_iter);
    targets_len += len_diffs[min_idx];

    auto pred_n_gram = count_ngram(tokenized_preds[i], n_gram);

    std::unordered_map<std::string, size_t> targets_n_gram;
    for (const auto& tokenized_target : tokenized_targets[i]) {
      auto target_n_gram = count_ngram(tokenized_target, n_gram);
      for (const auto& [k, v] : target_n_gram) {
        if (targets_n_gram.count(k)) {
          targets_n_gram[k] = std::max(targets_n_gram[k], v);
        } else {
          targets_n_gram[k] = v;
        }
      }
    }

    std::unordered_map<std::string, size_t> n_gram;
    for (const auto& [k, v] : targets_n_gram) {
      if (pred_n_gram.count(k))
        n_gram[k] = std::min(pred_n_gram[k], v);
    }
    for (const auto& [k, v] : pred_n_gram) {
      if (targets_n_gram.count(k))
        n_gram[k] = std::min(targets_n_gram[k], v);
    }

    for (const auto& [k, v] : n_gram) {
      size_t k_n_gram = std::count(k.begin(), k.end(), '|');
      num[k_n_gram - 1] += v;
    }
    for (const auto& [k, v] : pred_n_gram) {
      size_t k_n_gram = std::count(k.begin(), k.end(), '|');
      denom[k_n_gram - 1] += v;
    }
  }

  et = std::chrono::high_resolution_clock::now();
  ms = et - st;
  std::cout << fmt::format("Ngrams: {} s", ms.count()) << std::endl;

  st = std::chrono::high_resolution_clock::now();

  std::vector<double> precision(n_gram);
  if (smooth) {
    std::transform(num.begin(), num.end(), denom.begin(), precision.begin(),
                   [](const auto& n_val, const auto& denom_val) {
                     return static_cast<double>((n_val + 1)) / (denom_val + 1);
                   });
    precision[0] = static_cast<double>(num[0]) / denom[0];
  } else {
    std::transform(num.begin(), num.end(), denom.begin(), precision.begin(),
                   [](const auto& n_val, const auto& denom_val) {
                     return static_cast<double>(n_val) / denom_val;
                   });
  }

  std::transform(precision.begin(), precision.end(), weights.begin(),
                 precision.begin(), [](auto& precision_val, auto& weight_val) {
                   return weight_val * std::log(precision_val);
                 });

  double geometric_mean = exp(std::reduce(precision.begin(), precision.end()));
  double brevity_penalty =
      preds_len > targets_len
          ? 1
          : exp(1 - (static_cast<double>(targets_len) / preds_len));
  double bleu = brevity_penalty * geometric_mean;

  et = std::chrono::high_resolution_clock::now();
  ms = et - st;
  std::cout << fmt::format("Bleu: {} s", ms.count()) << std::endl;
  return bleu;
}

auto run_torchmetrics_example() -> void {
  std::vector<std::string> preds = {"the squirrel is eating the nut",
                                    "the cat is on the mat"};
  std::vector<std::vector<std::string>> references = {
      {"a squirrel is eating a nut", "the squirrel is eating a tasty nut"},
      {"there is a cat on the mat", "a cat is on the mat"}};

  std::cout << bleu_score(preds, references) << std::endl;
}

auto run_synthetic_example() -> void {
  auto n_tokens{1000};
  auto n_samples{100};

  std::string _token{"foo "};
  std::string s;
  s.reserve(_token.size() * n_tokens);
  while (n_tokens--)
    s += _token;

  std::vector<std::string> preds(n_samples, s);
  std::vector<std::vector<std::string>> targets(
      n_samples, std::vector<std::string>(n_samples, s));

  bleu_score(preds, targets);
}

int main() {
  run_torchmetrics_example();
  // run_synthetic_example();
}
