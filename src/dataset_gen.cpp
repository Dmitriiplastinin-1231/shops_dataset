#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <iomanip>
#include <cmath>

#include <vector>
#include <string>
#include <utility>
#include <map>

#include "dataset_gen.hpp"

const char VISA = '4';
const char MASTERCARD = '5';
const char MIR = '2';

int TOTAL_ROWS = 100000;
const int MIN_CATEGORIES = 50;
const int MIN_BRANDS = 500;
const int MAX_CARD_REPEATS = 5;
const int MIN_ITEMS_PER_RECEIPT = 2;
const int MAX_ITEMS_PER_RECEIPT = 2;
// веса для карт
int WEIGHT_OF_MASTERCARD = 20;
int WEIGHT_OF_MIR = 20;
// const int WEIGHT_OF_VISA = 20;

int WEIGHT_OF_SBER = 20;
int WEIGHT_OF_TBANK = 20;
int WEIGHT_OF_VTB = 20;

const std::vector<std::string> CATEGORY_CLASSES = {"техника", "продукты", "для дома и офиса", "спорт", "косметические средства", "для детей"};
// const std::vector<std::string> CATEGORY_CLASSES = {"техника", "продукты", "для дома", "спорт", "косметические средства",  "товары для учебы"};

// const bool TECHN = true;
// const bool PRODUCT = true;
// // предметы интерьера
// const bool INTER = true;
// const bool FOR_HOUSE = true;
// const bool CLOTHES = true;
// const bool SPORT = true;
// const bool KOSMETIC = true;
// const bool FOR_CHILD = true;
// const bool STUDY = true;


int main()
{
    std::cout << "Введите веса платежных систем, через пробелы в след. порядке: MasterCard, Мир: " << std::endl;
    std::cin >> WEIGHT_OF_MASTERCARD >> WEIGHT_OF_MIR;
    std::cout << "Введите веса банков, через пробелы в след. порядке: SberBank, Tbank, VTB: " << std::endl;
    std::cin >> WEIGHT_OF_SBER >> WEIGHT_OF_TBANK >> WEIGHT_OF_VTB;
    std::cout << "Введите количество строк в датасете: " << std::endl;
    std::cin >> TOTAL_ROWS;
    std::cout << "Генерация датасета покупок в Санкт-Петербурге..." << std::endl;

    try 
    {
        if (WEIGHT_OF_MASTERCARD + WEIGHT_OF_MIR <= 0) 
        {
            throw std::runtime_error("Некорректные веса платежных систем");
        } else if (WEIGHT_OF_SBER + WEIGHT_OF_TBANK + WEIGHT_OF_VTB <= 0)
        {
            throw std::runtime_error("Некорректные веса банков");
        } else if (TOTAL_ROWS <= 0)
        {
            throw std::runtime_error("некорректное кол-во строк");
        }

        DatasetGenerator generator;
        generator.generateDataset("spb_purchases_dataset.csv");
    } catch (const std::exception& e)
    {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}




int RandomGenerator::randomInt(int min, int max) {
    return std::uniform_int_distribution<int>(min, max)(gen);
}

double RandomGenerator::randomDouble(double min, double max) {
    return std::uniform_real_distribution<double>(min, max)(gen);
}

template<typename T>
const T& RandomGenerator::randomChoice(const std::vector<T>& vec) {
    if (vec.empty()) {
        throw std::runtime_error("Empty vector in randomChoice");
    }
    return vec[randomInt(0, vec.size() - 1)];
}





DatasetGenerator::DatasetGenerator()
{
    initialize_dictionaries();
}

void DatasetGenerator::initialize_dictionaries()
{
    std::cout << "Инилизация словарей..." << std::endl;

    parse_store_from_csv("stores.csv");
    parse_brands_from_csv("category.csv");

}

void DatasetGenerator::parse_store_from_csv(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Ошибка открытия файла: " << filename << std::endl;
        return;
    }
    
    std::string line;
    std::getline(file, line);

    while (std::getline(file, line))
    {
        size_t pos = 0;
        std::string modified_line = line;
        
        while ((pos = modified_line.find(",\"", pos)) != std::string::npos) {
            modified_line.replace(pos, 1, "|");
        }
       
        if ((pos = modified_line.rfind("\",")) != std::string::npos){
            modified_line.replace(pos, 2, "\"|");
        }


        std::string store_name, categories_str, coordinates_str, store_category;
        std::stringstream ss(modified_line);
        std::getline(ss, store_name, '|');
        std::getline(ss, categories_str, '|');
        std::getline(ss, coordinates_str, '|');
        std::getline(ss, store_category);

        Store store;
        store.name = store_name;
        store.working_hours = "10:00-22:00";
        
        
        store.categories_names = parse_str_in_vector(categories_str, ',');
        for (std::string category : store.categories_names)
        {
            
            if (std::find(all_categories_names.begin(), all_categories_names.end(), category) == all_categories_names.end())
            {
                Category* new_category = new Category;
                new_category->name = category; 
                all_categories_names.push_back(category);
                all_categories.push_back(new_category);
                store.categories.push_back(new_category);
            } else 
            {
                for (Category* category_obj : all_categories)
                {
                    if (category_obj->name == category) 
                    { 
                        store.categories.push_back(category_obj);
                        break;
                    }
                }
            }
        }
        

        std::vector coordinates = parse_str_in_vector(coordinates_str, ';');
        for (std::string coordination : coordinates)
        {
            std::stringstream coord_stream(coordination);
            std::string lat_str, lon_str;

            if (std::getline(coord_stream, lat_str, ',') && std::getline(coord_stream, lon_str)) 
            {
                try {
                    double lat = std::stod(lat_str);
                    double lon = std::stod(lon_str);
                    store.coordinates.push_back({lat, lon});
                } catch (const std::exception& e)
                {
                    std::cerr << "Ошибка парсинга координат для магазина " << store_name 
                              << ": " << e.what() << " в строке: " << coordination << std::endl;
                }
            }
        }

        stores.push_back(store);
    }

    

    file.close();
    std::cout << "Загружено магазинов: " << stores.size() << std::endl;
    std::cout << "Уникальных категорий: " << all_categories_names.size() << std::endl;
}

void DatasetGenerator::parse_brands_from_csv(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Ошибка открытия файла: " << filename << std::endl;
        return;
    }

    std::string line;
    
    while (std::getline(file, line))
    {
        size_t pos = 0;
        std::string modified_line = line;
        while ((pos = modified_line.find("\",", pos)) != std::string::npos) {
            modified_line.replace(pos+1, 1, "|");
        }



        std::stringstream ss(modified_line);
        
        std::string category_name, brands_str, price, category_class;
        std::getline(ss, category_name, ',');
        std::getline(ss, brands_str, '|');
        std::getline(ss, price, ',');
        std::getline(ss, category_class);


        bool is_not_true_categ_class = false;
        if (std::find(CATEGORY_CLASSES.begin(), CATEGORY_CLASSES.end(), category_class) == CATEGORY_CLASSES.end()) 
        {
            continue;
        }

        Category* this_category;
        for (Category* category : all_categories)
        {
            if (category->name == category_name)
            {
                this_category = category;
                break;
            }
        }

        try 
        {
            this_category->brands = parse_str_in_vector(brands_str, ',');
            this_category->price = std::stoi(price);
            this_category->category_class = category_class;

            for (std::string brand : this_category->brands)
            {
                
                if (std::find(all_brands_names.begin(), all_brands_names.end(), brand) == all_brands_names.end())
                {
                    all_brands_names.push_back(brand);
                }
            }

        } catch (const std::exception& e)
        {
            std::cerr << "Ошибка при обработке категории: " << category_name << std::endl;
        }
    }


    file.close();
    std::cout << "Загружено уникальных брендов:" << all_brands_names.size() << std::endl;
}

std::vector<std::string> DatasetGenerator::parse_str_in_vector(const std::string& str, const char ch)
{
    // if ch = ','
    // str = "\"text, text, text, ... , text\""
    std::stringstream ss(str);

    std::string word;
    std::getline(ss, word, '"');

    std::vector<std::string> vect;
    try
    {
        while (std::getline(ss, word, ch)) vect.push_back(word);
    } catch (const std::exception& e)
    {
        std::cerr << "Ошибка перевода строки в вектор на строке: " << str << std::endl << std::endl;
    }

    std::string end_word (vect[vect.size() - 1].begin(), vect[vect.size() - 1].end() - 1);
    vect[vect.size() - 1] = end_word;

    return vect;
}

std::string DatasetGenerator::generate_date_time(const Store& store) 
{
    int month = rng.randomInt(1, 12);
    int day = rng.randomInt(1, 28);
    int hour = rng.randomInt(10, 21);
    int minute = rng.randomInt(0, 59);
    
    std::stringstream ss;
    ss << "2024-" << std::setw(2) << std::setfill('0') << month << "-"
       << std::setw(2) << std::setfill('0') << day << "T"
       << std::setw(2) << std::setfill('0') << hour << ":"
       << std::setw(2) << std::setfill('0') << minute << "+03:00";
    return ss.str();
}

std::pair<double, double> DatasetGenerator::get_random_coordinate(const Store& store) 
{
    if (store.coordinates.empty()) {
        return {59.9311, 30.3609};
    }
    return store.coordinates[rng.randomInt(0, store.coordinates.size() - 1)];
}

std::string DatasetGenerator::format_coordinates(double lat, double lon)
{
    double roundedLat = std::round(lat * 100000000) / 100000000;
    double roundedLon = std::round(lon * 100000000) / 100000000;
        
    std::stringstream ss;
    ss << std::fixed << std::setprecision(8) << "\"" << roundedLat << "," << roundedLon << "\"";
    return ss.str();
}

std::vector<Product> DatasetGenerator::generate_products_for_store(const Store& store, int count) 
{
    std::vector<Product> products;
        
    for (int i = 0; i < count; i++) {
        Product product;
            
        product.category = rng.randomChoice(store.categories_names);
            
        for (Category* category : all_categories)
        {
            if ((category->name == product.category) && (!category->brands.empty()))
            {
                product.brand = rng.randomChoice(category->brands);
                product.price = category->price;
            }
        }
            
            
        products.push_back(product);
    }
        
    return products;
}

std::string DatasetGenerator::escape_csv_field(const std::string& field) 
{
    if (field.find(',') != std::string::npos || 
        field.find('"') != std::string::npos || 
        field.find('\n') != std::string::npos) {
        std::string escaped = field;
        
        size_t pos = 0;
        while ((pos = escaped.find('"', pos)) != std::string::npos) {
            escaped.replace(pos, 1, "\"\"");
            pos += 2;
        }
        return "\"" + escaped + "\"";
    }
    return field;
}

std::string DatasetGenerator::generate_card_number(char first_num) 
{
    std::stringstream ss;
    ss << first_num << std::setw(3) << std::setfill('0') << rng.randomInt(0, 9999);
    for (int i = 0; i < 3; i++) {
        ss << " ";
        ss << std::setw(4) << std::setfill('0') << rng.randomInt(0, 9999);
    }
    return ss.str();
}

char DatasetGenerator::generate_card_pay_system()
{
    // int random_change = rng.randomInt(1, WEIGHT_OF_MASTERCARD + WEIGHT_OF_MIR + WEIGHT_OF_VISA);
    int random_change = rng.randomInt(1, WEIGHT_OF_MASTERCARD + WEIGHT_OF_MIR);
    
    if (random_change <= WEIGHT_OF_MASTERCARD){return MASTERCARD;}
    else if (random_change <= WEIGHT_OF_MASTERCARD + WEIGHT_OF_MIR){return MIR;}
    // else if (random_change <= WEIGHT_OF_MASTERCARD + WEIGHT_OF_MIR + WEIGHT_OF_VISA){return VISA;}
    return ' ';
}

void DatasetGenerator::generate_card_bank(std::string card_num)
{
    int random_change = rng.randomInt(1, WEIGHT_OF_SBER + WEIGHT_OF_TBANK + WEIGHT_OF_VTB);

    if (random_change <= WEIGHT_OF_SBER){card_bank[card_num] = "Sberbank";}
    else if (random_change <= WEIGHT_OF_TBANK){card_bank[card_num] = "TBank";}
    else if (random_change <= WEIGHT_OF_VTB){card_bank[card_num] = "VTB";}
    
}

std::string DatasetGenerator::get_valid_card() {
    if (!used_cards.empty() && rng.randomDouble(0, 1) < 0.3) {
        std::string card = used_cards[rng.randomInt(0, used_cards.size() - 1)];
        if (card_usage_count[card] < MAX_CARD_REPEATS) {
            return card;
        }
    }
    
    std::string new_card = generate_card_number(generate_card_pay_system());
    used_cards.push_back(new_card); 
    // generate_card_bank(new_card);
    card_usage_count[new_card] = 0;
    return new_card;
}

template<typename T>
std::string DatasetGenerator::join_list(const std::vector<T>& items) {
    if (items.empty()) return "";
    
    std::stringstream ss;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i > 0) ss << ",";
        ss << items[i];
    }
    return ss.str();
}
    
std::string DatasetGenerator::join_list(const std::vector<double>& prices) {
    if (prices.empty()) return "";
    
    std::stringstream ss;
    for (size_t i = 0; i < prices.size(); ++i) {
        if (i > 0) ss << ",";
        ss << std::fixed << std::setprecision(2) << prices[i];
    }
    return ss.str();
}

std::string DatasetGenerator::generate_receipt_number(const std::string& store_name) {
    std::string receipt;
    do {
        receipt = std::to_string(rng.randomInt(1000, 99999));
    } while (store_receipts[store_name].count(receipt) > 0);
    
    store_receipts[store_name].insert(receipt);
    return receipt;
}

void DatasetGenerator::generateDataset(const std::string& filename)
{
    std::cout << "Начало генерации датасета..." << std::endl;

    std::ofstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Ошибка открытия файла!" << std::endl;
        return;
    }

    file << "store_name,datetime,coordinates,categories,brands,item_prices,"
             << "card_number,items_count,receipt_number,total_price\n";

    int rows_generated = 0;

    while (rows_generated < TOTAL_ROWS)
    {
        Store& store = stores[rng.randomInt(0, stores.size() - 1)];
        bool is_empty_brands = true;
        for (Category* category : store.categories)
        {
            if (!category->brands.empty()) is_empty_brands = false;
        }
        
        if (store.categories_names.empty() || store.coordinates.empty() || is_empty_brands) continue;
        
        std::string datetime = generate_date_time(store);
        std::pair<double, double> coords = get_random_coordinate(store);
        std::string coordinates = format_coordinates(coords.first, coords.second);
        
        int items_count = rng.randomInt(MIN_ITEMS_PER_RECEIPT, MAX_ITEMS_PER_RECEIPT);
        std::vector<Product> products = generate_products_for_store(store, items_count);
        
        if (products.empty()) continue;
        
        std::vector<std::string> product_categories;
        std::vector<std::string> product_brands;
        std::vector<double> product_prices;
        double total_price = 0.0;
        
        for (const Product& product : products) 
        {
            product_categories.push_back(product.category);
            product_brands.push_back(product.brand);
            product_prices.push_back(product.price);
            total_price += product.price;
        }
        
        // std::string categories_str = escape_csv_field(join_list(product_categories));
        // std::string brands_str = escape_csv_field(join_list(product_brands));
        // std::string prices_str = escape_csv_field(join_list(product_prices));
        
        std::string card_number = get_valid_card();
        card_usage_count[card_number]++;
        
        std::string receipt_number = generate_receipt_number(store.name);
        
        for (int i = 0; i < items_count; i++)
        {
            
            file << escape_csv_field(store.name) << ","
                     << escape_csv_field(datetime) << ","
                     << coordinates << ","
                     << product_categories[i] << ","
                     << product_brands[i] << ","
                     << product_prices[i] << ","
                     << escape_csv_field(card_number) << ","
                     << items_count << ","
                     << escape_csv_field(receipt_number) << ","
                     << std::fixed << std::setprecision(2) << total_price << "\n";

        }
            
        rows_generated += items_count;

        if (rows_generated % 5000 == 0) 
        {
            std::cout << "Сгенерировано строк: " << rows_generated << "/" << TOTAL_ROWS << std::endl;
        }
    }
    
    std::cout << "Сгенерировано всего строк: " << rows_generated << "/" << TOTAL_ROWS << std::endl;
    

    file.close();
}