#include "main_window_anonym.hpp"

#include "dataset_anonim.hpp"
#include <QVBoxLayout>
#include <QLabel>
#include <QFileDialog>
#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    setupUI();
}

void MainWindow::setupUI() {
    QWidget *central = new QWidget(this);
    QVBoxLayout *layout = new QVBoxLayout(central);

    storeNameBox = new QCheckBox("STORE_NAME", this);
    dateBox = new QCheckBox("DATE", this);
    coordBox = new QCheckBox("COORDINATES", this);
    catBox = new QCheckBox("CATEGORIES", this);
    brandBox = new QCheckBox("BRANDS", this);
    itemPriceBox = new QCheckBox("ITEM_PRICE", this);
    cardBox = new QCheckBox("CARD_NUMBER", this);
    receiptBox = new QCheckBox("RECEIPT_NUMBER", this);
    totalPriceBox = new QCheckBox("TOTAL_PRICE", this);

    QList<QCheckBox*> boxes = {storeNameBox, dateBox, coordBox, catBox, brandBox,
                               itemPriceBox, cardBox, receiptBox, totalPriceBox};
    for (auto *b : boxes) b->setChecked(true);

    sourceEdit = new QLineEdit("spb_purchases_dataset.csv", this);
    outputEdit = new QLineEdit("spb_purchases_anonym_dataset.csv", this);
    k_anonymity_file_edit = new QLineEdit("spb_purchases_anonym_dataset.csv", this);

    QPushButton *anonBtn = new QPushButton("Обезличить", this);
    QPushButton *kanonBtn = new QPushButton("Рассчитать k-anonymity", this);
    logArea = new QTextEdit(this);
    logArea->setReadOnly(true);

    layout->addWidget(new QLabel("Исходный файл:"));
    layout->addWidget(sourceEdit);
    layout->addWidget(new QLabel("Файл вывода:"));
    layout->addWidget(outputEdit);
    layout->addWidget(new QLabel("Файл для расчета k-anonymity:"));
    layout->addWidget(k_anonymity_file_edit);
    layout->addWidget(new QLabel("Выберите квази-идентификаторы:"));
    for (auto *b : boxes) layout->addWidget(b);
    layout->addWidget(anonBtn);
    layout->addWidget(kanonBtn);
    layout->addWidget(new QLabel("Лог:"));
    layout->addWidget(logArea);

    connect(anonBtn, &QPushButton::clicked, this, &MainWindow::runAnonymization);
    connect(kanonBtn, &QPushButton::clicked, this, &MainWindow::calculateKAnonymity);

    setCentralWidget(central);
}

void MainWindow::runAnonymization() {
    QString src = sourceEdit->text();
    QString dst = outputEdit->text();

    STORE_NAME   = storeNameBox->isChecked();
    DATE         = dateBox->isChecked();
    COORDINATES  = coordBox->isChecked();
    CATEGORIES   = catBox->isChecked();
    BRANDS       = brandBox->isChecked();
    ITEM_PRICE   = itemPriceBox->isChecked();
    CARD_NUMBER  = cardBox->isChecked();
    RECEIPT_NUMBER = receiptBox->isChecked();
    TOTAL_PRICE  = totalPriceBox->isChecked();

    logMessage("Запуск обезличивания...");
    anonymization(src.toStdString(), dst.toStdString());
    logMessage("✅ Обезличивание завершено");
}

void MainWindow::calculateKAnonymity() {
    QString src = k_anonymity_file_edit->text();
    logMessage("Подсчет k-anonymity...");
    // тут можно добавить улучшенную функцию, возвращающую статистику
    check_k_anonym(src.toStdString(), true);


    logMessage("⚠️Топ 5 худших k-anonymity: ");
    int counter = 0;
    for (const auto& [k, count] : k_anonym) {
        logMessage(QString::fromStdString(std::to_string(k) + ": " + std::to_string(count) + "(" + std::to_string((static_cast<float>(count)/rows_in_dataset)*100) + ")"));
        counter++;
        if (counter == 5) break;
    }


    logMessage("✅ k-anonymity рассчитано");
}

void MainWindow::logMessage(const QString &msg) {
    logArea->append(msg);
}
