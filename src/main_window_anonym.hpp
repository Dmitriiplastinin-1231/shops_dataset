#pragma once
#include <QMainWindow>
#include <QCheckBox>
#include <QTextEdit>
#include <QPushButton>
#include <QLineEdit>

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override = default;

private slots:
    void runAnonymization();
    void calculateKAnonymity();

private:
    void setupUI();
    void logMessage(const QString &msg);

    QCheckBox *storeNameBox, *dateBox, *coordBox, *catBox, *brandBox;
    QCheckBox *itemPriceBox, *cardBox, *receiptBox, *totalPriceBox;
    QLineEdit *sourceEdit, *outputEdit, *k_anonymity_file_edit;
    QTextEdit *logArea;
};
