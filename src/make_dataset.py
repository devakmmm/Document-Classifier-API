from __future__ import annotations
import os
import random
import pandas as pd

OUT_PATH = "data/raw/docs.csv"

INVOICE_TEMPLATES = [
    "Invoice #{n} Total Due ${amt}. Payment due within {days} days.",
    "Bill To: Customer {n}. Amount Due ${amt}. Invoice Date {date}.",
    "Invoice Number {n}. Subtotal ${amt}. Please remit payment.",
]

RECEIPT_TEMPLATES = [
    "Receipt #{n} Thank you for your purchase. Total ${amt}.",
    "Transaction ID {n}. Paid ${amt} via credit card.",
    "Store receipt. Order {n}. Amount ${amt}.",
]

CONTRACT_TEMPLATES = [
    "This agreement is made on {date} between Party A and Party B.",
    "Contract effective {date}. Terms and conditions apply.",
    "This legally binding agreement outlines obligations of both parties.",
]

ID_TEMPLATES = [
    "Driver License Number D{n}. Date of Birth {date}.",
    "Identification Card ID-{n}. Issued on {date}.",
    "Government issued ID number {n}.",
]

def gen_samples(templates, label, count):
    rows = []
    for _ in range(count):
        rows.append({
            "text": random.choice(templates).format(
                n=random.randint(1000, 9999),
                amt=round(random.uniform(5, 5000), 2),
                days=random.choice([15, 30, 45]),
                date=random.choice(["01/12/2023", "05/03/2024", "11/18/2022"])
            ),
            "label": label
        })
    return rows

def main():
    os.makedirs("data/raw", exist_ok=True)

    rows = []
    rows += gen_samples(INVOICE_TEMPLATES, "invoice", 150)
    rows += gen_samples(RECEIPT_TEMPLATES, "receipt", 150)
    rows += gen_samples(CONTRACT_TEMPLATES, "contract", 120)
    rows += gen_samples(ID_TEMPLATES, "id", 120)

    random.shuffle(rows)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_PATH, index=False)

    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print(df["label"].value_counts())

if __name__ == "__main__":
    main()
