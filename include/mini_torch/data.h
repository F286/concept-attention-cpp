#pragma once
#include <vector>
#include <cstddef>
#include <algorithm>
#include <iterator>
#include <ranges>
#include <random>
#include <numeric>
#include <fstream>
#include <string>

/**
 * @brief Lightweight container of samples accessible by index.
 * Mimics a subset of std::vector providing const iteration and indexing.
 */
template<typename T>
class Dataset {
public:
    using value_type = T; ///< element type
    /// @brief Construct dataset from samples
    explicit Dataset(std::vector<T> samples) : m_samples(std::move(samples)) {}
    /// @brief Iterator to first sample
    auto begin() const { return m_samples.begin(); }
    /// @brief Iterator past last sample
    auto end() const { return m_samples.end(); }
    /// @brief Number of samples
    size_t size() const { return m_samples.size(); }
    /// @brief Access sample by index
    const T &operator[](size_t idx) const { return m_samples[idx]; }
private:
    std::vector<T> m_samples; ///< stored samples
};

/// @brief Load text file into character dataset
inline Dataset<char> load_text_dataset(const std::string &path) {
    std::ifstream file(path);
    std::vector<char> data((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
    return Dataset<char>(std::move(data));
}

/**
 * @brief Iterable view producing mini-batches from a Dataset.
 * Provides an input_range yielding std::vector batches similar to
 * torch::data::DataLoader.
 */
template<typename D>
class DataLoader {
public:
    using dataset_type = D; ///< underlying dataset type
    using value_type = typename D::value_type; ///< sample type
    using batch_type = std::vector<value_type>; ///< batch vector type

    /**
     * @brief Forward iterator over mini-batches
     * Dereferences to a batch_type containing a slice of the dataset.
     */
    class iterator {
    public:
        using iterator_category = std::input_iterator_tag; ///< category tag
        using value_type = batch_type; ///< iterator value type
        using difference_type = std::ptrdiff_t; ///< difference type
        /// @brief Default construct null iterator
        iterator() = default;
        /// @brief Construct iterator for loader and batch index
        iterator(const DataLoader *loader, size_t pos)
            : m_loader(loader), m_pos(pos) {}
        /// @brief Produce batch at current position
        batch_type operator*() const {
            batch_type batch;
            size_t start = m_pos * m_loader->m_batch_size;
            size_t end = std::min(start + m_loader->m_batch_size,
                                  m_loader->m_order.size());
            batch.reserve(end - start);
            for (size_t i = start; i < end; ++i)
                batch.push_back((*m_loader->m_dataset)[m_loader->m_order[i]]);
            return batch;
        }
        /// @brief Advance to next batch
        iterator &operator++() { ++m_pos; return *this; }
        /// @brief Post-increment
        iterator operator++(int) { auto tmp = *this; ++*this; return tmp; }
        /// @brief Compare iterator equality
        bool operator==(const iterator &other) const {
            return m_pos == other.m_pos && m_loader == other.m_loader;
        }
        /// @brief Negated equality
        bool operator!=(const iterator &other) const { return !(*this == other); }
    private:
        const DataLoader *m_loader; ///< parent loader
        size_t m_pos; ///< batch index
    };

    /// @brief Construct loader referencing dataset
    DataLoader(const D &dataset, size_t batch_size, bool shuffle = false)
        : m_dataset(&dataset), m_batch_size(batch_size) {
        m_order.resize(dataset.size());
        std::iota(m_order.begin(), m_order.end(), 0);
        if (shuffle) {
            std::mt19937 rng(std::random_device{}());
            std::shuffle(m_order.begin(), m_order.end(), rng);
        }
    }
    /// @brief Iterator to first batch
    iterator begin() const { return iterator(this, 0); }
    /// @brief Iterator past last batch
    iterator end() const { return iterator(this, size()); }
    /// @brief Number of batches
    size_t size() const {
        return (m_order.size() + m_batch_size - 1) / m_batch_size;
    }
private:
    const D *m_dataset; ///< referenced dataset
    size_t m_batch_size; ///< batch size
    std::vector<size_t> m_order; ///< sample ordering
};

static_assert(std::ranges::range<Dataset<int>>); ///< dataset concept check
static_assert(std::ranges::input_range<DataLoader<Dataset<int>>>); ///< dataloader concept check

