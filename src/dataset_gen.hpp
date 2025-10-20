#pragma once

#include <random>

#include <vector>
#include <string>
#include <utility>
#include <map>
#include <unordered_set>


class RandomGenerator {
private:
    std::mt19937 gen;
    
public:
    RandomGenerator() : gen(std::random_device{}()) {}
    
    int randomInt(int min, int max);
    
    double randomDouble(double min, double max);

    template<typename T>
    const T& randomChoice(const std::vector<T>& vec);
};

struct Category 
{
    std::string name;
    int price;
    std::string category_class;
    std::vector<std::string> brands;
};

struct Store 
{
    std::string name;
    std::vector<std::pair<double, double>> coordinates;
    std::vector<Category*> categories;
    std::vector<std::string> categories_names;
    std::string working_hours;
};


struct Product 
{
    std::string category;
    std::string brand;
    double price;
};

class DatasetGenerator 
{
private:
    RandomGenerator rng;
    std::vector<Store> stores;
    std::vector<Category*> all_categories;
    std::vector<std::string> all_categories_names;
    std::vector<std::string> all_brands_names;
    std::map<std::string, int> card_usage_count;
    std::vector<std::string> used_cards;
    std::map<std::string, std::unordered_set<std::string>> store_receipts;
public:
    DatasetGenerator();
private:
    void parse_store_from_csv(const std::string& filename);
    void parse_brands_from_csv (const std::string& filename);
    std::string escape_csv_field(const std::string& field);
    std::vector<std::string> parse_str_in_vector(const std::string& str, const char ch);
    
    void initialize_brands();
    void initialize_dictionaries();
    void initialize_brand_categories_and_prices();
    
    double get_base_price(const std::string& category, const std::string& brand);
    std::vector<std::string> get_unique_categories(const std::vector<Product>& products);
    std::pair<double, double> get_random_coordinate(const Store& store);
    
    std::string generate_card_number();
    std::string generate_date_time(const Store& store);
    std::string generate_receipt_number(const std::string& store_name);
    std::vector<Product> generate_products_for_store(const Store& store, int count);
    
    std::string format_coordinates(double lat, double lon);
    std::string get_valid_card();

    template<typename T>
    std::string join_list(const std::vector<T>& items);
    std::string join_list(const std::vector<double>& items);
public:
    void generateDataset(const std::string& filename);
};