#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>


#include <string>
#include <map>
#include <unordered_map>
#include <vector>


const char VISA = '4';
const char MASTERCARD = '5';
const char MIR = '2';

const bool STORE_NAME = true;
const bool DATE = true;
const bool COORDINATES = true;
const bool CATEGORIES = true;
const bool BRANDS = true;
const bool ITEM_PRICE = true;
const bool CARD_NUMBER = true;
const bool RECEIPT_NUMBER = true;
const bool TOTAL_PRICE = true;

std::unordered_map<std::string, std::string> categories;
std::unordered_map<std::string, std::string> brands;
std::unordered_map<std::string, std::string> store_categories;

void check_k_anonym(std::string file_name, bool is_input);
void anonymization(std::string file_name, std::string anonym_file_name);
void parse_categories_and_brands(std::string file_name);
void parse_brands_country_from_csv(std::string file_name);
void parse_store_category_from_csv(std::string file_name);

std::string store_name_anonymization(std::string store_name);
std::string escape_csv_field(const std::string& field);
std::string date_anonymization(std::string date);
std::string coordinates_anonymization(std::string coordinates);
std::string price_anonymization(std::string price_str, int small, int normal);
std::string price_anonymization(std::string price_str, int middle);
std::string card_anonymization(std::string card_number);
std::string categories_anonymization(std::string category);
std::string brand_anonymization(std::string brand);


int main()
{
    std::string source_file = "spb_purchases_dataset.csv";
    std::string output_file = "spb_purchases_anonym_dataset.csv";

    std::cout << "Заполнение ассоциативных массивов..." << std::endl;
    parse_store_category_from_csv("stores.csv");
    std::cout << "1) Имена магазина -> категория магазина." << std::endl; 
    parse_brands_country_from_csv("brands_country.csv");
    std::cout << "2) Бренды -> страна этого бренда." << std::endl; 
    parse_categories_and_brands("category.csv");
    std::cout << "3) Категории -> обобщенный класс продукта." << std::endl; 
    
    std::cout << "Вывод k-anonymity синтетических данных..." << std::endl;
    check_k_anonym(source_file, false);
    anonymization(source_file, output_file);
    std::cout << "Вывод k-anonymity обезличенного набора данных..." << std::endl;
    check_k_anonym(output_file, true);
}

void parse_store_category_from_csv(std::string file_name)
{
    std::ifstream file(file_name);
    if (!file.is_open())
    {
        std::cerr << "Ошибка открытия файла: " << file_name << std::endl;
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
        
        if (store_categories.find(store_name) == store_categories.end())
        {
            store_categories[store_name] = store_category;
        }
    }

    file.close();
}


void parse_brands_country_from_csv(std::string file_name)
{
    std::ifstream file(file_name);
    if(!file.is_open())
    {
        std::cerr << "Ошибка открытия файла: " << file_name << std::endl;
    } 

    std::string line;
    while(std::getline(file, line))
    {
        std::stringstream ss(line);

        std::string brand, country;
        std::getline(ss, brand, ',');
        std::getline(ss, country);

        if (brands.find(brand) == brands.end())
        {
            brands[brand] = country;
        }
    }


    file.close();
}
void parse_categories_and_brands(std::string file_name)
{
    std::ifstream file(file_name);
    if (!file.is_open())
    {
        std::cerr << "Ошибка открытия файла: " << file_name << std::endl;
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

        
        if (categories.find(category_name) == categories.end())
        {
            categories[category_name] = category_class;
        }
        
        
    }

    file.close();
}

void check_k_anonym(std::string file_name, bool is_input)
{
    std::ifstream file(file_name);

    if (!file.is_open())
    {
        std::cerr << "Ошибка открытия файла: " << file_name << ". Для подсчета k-anonymity." << std::endl;
        return;
    }

    std::unordered_map<std::string, int> line_counts;
    std::string line;

    std::getline(file, line);

    while (std::getline(file, line)) {
        if (!line.empty()) {
            ++line_counts[line];
        }
    }
    file.close();

    
    std::map<int, int> k_distribution;

    for (const auto& [line_text, k] : line_counts) {
        if (is_input && k == 1)
        {
            std::cout << std::endl<< std::endl << line_text << std::endl<< std::endl;
        }
        k_distribution[k] += k;
    }

    
    std::cout << "{";
    bool first = true;
    for (const auto& [k, count] : k_distribution) {
        if (!first) std::cout << ", ";
        std::cout << k << ": " << count;
        first = false;
    }
    std::cout << "}" << std::endl;
}

std::string escape_csv_field(const std::string& field) 
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

void anonymization(std::string file_name, std::string anonym_file_name)
{
    std::cout << "Начало обезличивания..." << std::endl;

    std::ifstream file(file_name);
    std::ofstream anonym_file(anonym_file_name);

    if (!file.is_open())
    {
        std::cerr << "Ошибка открытия файла: " << file_name << std::endl;
        return;
    }

    if (!anonym_file.is_open())
    {
        std::cerr << "Ошибка открытия файла: " << anonym_file_name << std::endl;
        return;
    }

    anonym_file << "store_name,datetime,coordinates,categories,brands,item_prices,"
             << "card_number,items_count,receipt_number,total_price\n";

    std::string line;
    std::getline(file, line);

    while (std::getline(file, line)) {

        std::string store_name;
        std::string datetime;
        std::string coordinates;
        std::string category;
        std::string brand;
        std::string item_price;
        std::string card_number;
        std::string items_count;
        std::string receipt_number;
        std::string total_price;
        std::string trash;

        if (!line.empty()) {
            std::stringstream ss(line);
    

            std::getline(ss, store_name, ',');
            std::getline(ss, datetime, ',');
            
            std::getline(ss, trash, '"');
            std::getline(ss, coordinates, '"');
            std::getline(ss, trash, ',');

            std::getline(ss, category, ',');
            std::getline(ss, brand, ',');
            std::getline(ss, item_price, ',');
            std::getline(ss, card_number, ',');
            std::getline(ss, items_count, ',');
            std::getline(ss, receipt_number, ',');
            std::getline(ss, total_price);

            if (STORE_NAME) {store_name = store_name_anonymization(store_name);}
            if (DATE){datetime = date_anonymization(datetime);}
            if (COORDINATES){coordinates = coordinates_anonymization(coordinates);}
            if (CATEGORIES){category = categories_anonymization(category);}
            if (BRANDS){brand = brand_anonymization(brand);}
            if (ITEM_PRICE){item_price = price_anonymization(item_price, 500);}
            if (CARD_NUMBER){card_number = card_anonymization(card_number);}
            if (RECEIPT_NUMBER){receipt_number = "None";}
            if (TOTAL_PRICE){total_price = price_anonymization(total_price, 5000);}
        }
        
        anonym_file << escape_csv_field(store_name) << ","
            << escape_csv_field(datetime) << ","
            << coordinates << ","
            << category << ","
            << brand << ","
            << item_price << ","
            << escape_csv_field(card_number) << ","
            << items_count << ","
            << escape_csv_field(receipt_number) << ","
            << std::fixed << std::setprecision(2) << total_price << "\n";

    }

    

    anonym_file.close();
    file.close();

    std::cout << "Обезличивание произведено!" << std::endl;
}

std::string store_name_anonymization(std::string store_name) {return store_categories[store_name];}


// std::string date_anonymization(std::string date) {return date.substr(0, 7);}
std::string date_anonymization(std::string date) 
{
    int month = std::stoi(date.substr(5, 2));
    if (month < 3 || month == 12){return "Зима";}
    else if (month < 6){return "Весна";}
    else if (month < 9){return "Лето";}
    else if (month < 12){return "Осень";}

    return date.substr(5, 2);
}
std::string brand_anonymization(std::string brand) {return brands[brand];}
std::string coordinates_anonymization(std::string coordinates) 
{
    std::stringstream ss(coordinates);
    std::string lat, lon;

    std::getline(ss, lat, ',');
    std::getline(ss, lon);

    lat = lat.substr(0, 2);
    lon = lon.substr(0,2);
    
    return '"' + lat + ',' + lon + '"';
}

std::string price_anonymization(std::string price_str, int small, int normal){
    int price = std::stoi(price_str);

    if (price <= small) {return "Дешево";} 
    else if (price <= normal) {return "Нормальная";}
    else {return "Дорого";}
}
std::string price_anonymization(std::string price_str, int middle) {
    int price = std::stoi(price_str);

    if (price <= middle) {return ("до " + std::to_string(middle));} 
    else {return ("после " + std::to_string(middle));}
}

std::string card_anonymization(std::string card_number)
{
    if (card_number[0] == MASTERCARD) {return "MasterCard";}
    else if (card_number[0] == VISA) {return "VISA";}
    else if (card_number[0] == MIR) {return "МИР";}

    return "Неизвестная";
}

std::string categories_anonymization(std::string category ){return categories[category];}