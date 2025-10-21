#include <iostream>
#include <fstream>
#include <sstream>

#include <string>
#include <map>
#include <unordered_map>


const bool DATE = true;
const bool COORDINATES = true;
const bool CATEGORIES = true;
const bool BRANDS = true;
const bool ITEM_PRICE = true;
const bool CARD_NUMBER = true;
const bool ITEMS_COUNT = true;  
const bool RECEIPT_NUMBER = true;
const bool TOTAL_PRICE = true;


void check_k_anonym(std::string file_name);



void check_k_anonym(std::string file_name)
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

void anonymization()
{
    
}

int main()
{
    check_k_anonym("spb_purchases_dataset.csv");
}