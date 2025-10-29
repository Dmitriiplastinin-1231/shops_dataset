#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iomanip>


#include <string>
#include <map>
#include <unordered_map>
#include <vector>

extern bool STORE_NAME;
extern bool DATE;
extern bool COORDINATES;
extern bool CATEGORIES;
extern bool BRANDS;
extern bool ITEM_PRICE;
extern bool CARD_NUMBER;
extern bool RECEIPT_NUMBER;
extern bool TOTAL_PRICE;

extern std::map<int, int> k_anonym;
extern int rows_in_dataset;


void check_k_anonym(std::string file_name, bool is_input);
void anonymization(std::string file_name, std::string anonym_file_name);
void parse_categories_and_brands(std::string file_name);
void parse_brands_country_from_csv(std::string file_name);
void parse_store_category_from_csv(std::string file_name);

std::string store_name_anonymization(std::string store_name);
std::string escape_csv_field(const std::string& field);
std::string date_anonymization(std::string date);
std::string coordinates_anonymization(std::string coordinates);
std::string price_anonymization(std::string price_str, int small, int normal, int maximum);
std::string price_anonymization(std::string price_str, int middle);
std::string card_anonymization(std::string card_number);
std::string categories_anonymization(std::string category);
std::string brand_anonymization(std::string brand);