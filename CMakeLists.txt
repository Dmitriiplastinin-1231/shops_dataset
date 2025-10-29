cmake_minimum_required(VERSION 3.16)
project(DataAnonymizerGUI)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)



find_package(Qt6 REQUIRED COMPONENTS Widgets)

add_executable(DataAnonymizer
    src/main_anonym.cpp
    src/main_window_anonym.cpp
    src/main_window_anonym.hpp
    src/dataset_anonim.cpp
    src/dataset_anonim.hpp
)

target_link_libraries(DataAnonymizer PRIVATE Qt6::Widgets)

add_custom_command(TARGET DataAnonymizer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/spb_purchases_dataset.csv
        ${CMAKE_CURRENT_BINARY_DIR}/spb_purchases_dataset.csv
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_SOURCE_DIR}/spb_purchases_anonym_dataset.csv
        ${CMAKE_CURRENT_BINARY_DIR}/spb_purchases_anonym_dataset.csv
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/src/assets
        ${CMAKE_CURRENT_BINARY_DIR}/assets
)