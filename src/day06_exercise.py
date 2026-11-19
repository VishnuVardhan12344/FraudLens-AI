transactions = [
    {
        "transaction_id": "TX001",
        "amount": 500,
        "customer_average": 450,
        "transactions_today": 2,
        "international": False
    },
    {
        "transaction_id": "TX002",
        "amount": 6200,
        "customer_average": 700,
        "transactions_today": 14,
        "international": True
    },
    {
        "transaction_id": "TX003",
        "amount": 1800,
        "customer_average": 600,
        "transactions_today": 5,
        "international": False
    },
    {
        "transaction_id": "TX004",
        "amount": 9500,
        "customer_average": 900,
        "transactions_today": 18,
        "international": True
    },
    {
        "transaction_id": "TX005",
        "amount": 250,
        "customer_average": 500,
        "transactions_today": 1,
        "international": False
    }
]

for a in transactions:
    amount_ratio = a["amount"] / a["customer_average"]

    if amount_ratio > 5 and a["transactions_today"] > 10 and a["international"] is True:
        a["risk_level"] = "HIGH"
    elif amount_ratio > 5 or a["transactions_today"] > 7:
        a["risk_level"] = "MEDIUM"
    else:
        a["risk_level"] = "LOW"

    print(f"{a['transaction_id']} | ${a['amount']:.2f} | Ratio: {amount_ratio:.2f} | Risk: {a['risk_level']}")


high_count = 0
medium_count = 0
low_count = 0

for a in transactions:
    amount_ratio = a["amount"] / a["customer_average"]

    if amount_ratio > 5 and a["transactions_today"] > 10 and a["international"] is True:
        a["risk_level"] = "HIGH"
        high_count += 1
    elif amount_ratio > 5 or a["transactions_today"] > 7:
        a["risk_level"] = "MEDIUM"
        medium_count += 1
    else:
        a["risk_level"] = "LOW"
        low_count += 1

print(f"HIGH:{high_count}")
print(f"MEDIUM:{medium_count}")
print(f"LOW:{low_count}")       
total_amount = 0
for b in transactions:
    total_amount += b["amount"]
print(f"total_transaction_value: ${total_amount:.2f}")

international_count = 0 
for c in transactions:
    if c["international"] is True:
        international_count += 1
print(f"international_count: {international_count}")

largest_amount = 0
for d in transactions:
    if d["amount"] > largest_amount:
        largest_amount = d["amount"]
    
print(largest_amount)

print("========== FraudLens Batch Analysis ==========\n")
for a in transactions:
    amount_ratio = a["amount"] / a["customer_average"]
    total_amount += a["amount"]
    if a["amount"] > largest_amount:
        largest_amount = a["amount"]
    
    if c["international"] is True:
        international_count += 1


    if amount_ratio > 5 and a["transactions_today"] > 10 and a["international"] is True:
        a["risk_level"] = "HIGH"
        high_count += 1
    elif amount_ratio > 5 or a["transactions_today"] > 7:
        a["risk_level"] = "MEDIUM"
        medium_count += 1
    else:
        a["risk_level"] = "LOW"
        low_count += 1

    print(f"{a['transaction_id']} | ${a['amount']:.2f} | Ratio: {amount_ratio:.2f} | Risk: {a['risk_level']}")
print("\n--------------Summary---------------\n")
print(f"total_transaction: {len(transactions)}")
print(f"total_transaction_value: ${total_amount:.2f}")
print(f"international_count: {international_count}")
print(f"largest_transaction: {largest_amount}")



print(f"HIGH:{high_count}")
print(f"MEDIUM:{medium_count}")
print(f"LOW:{low_count}") 