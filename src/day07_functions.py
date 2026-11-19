##def means define a function.
def greet():
    print("Hi Vishnu")

greet()
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
def analyze(transaction_id):
    print(f" {transaction_id} is analyzed")

analyze("TX001")

def amount_ratio(amount, customer_average):
    ratio = amount / customer_average
    return ratio

result = amount_ratio(3000, 1000)
print(result)


for a in transactions:
    a_ratio = amount_ratio(a["amount"],a["customer_average"])
    print(a_ratio)

def determine_risk(amount_ratio, transactions_today, international):

    if amount_ratio > 5 and transactions_today > 10 and international:
        return "HIGH"

    elif amount_ratio > 3 or transactions_today > 7:
        return "MEDIUM"

    else:
        return "LOW"

a = determine_risk(8,13,True)
print(a)


def amount_ratio(transaction):
    amount = transaction["amount"]
    average = transaction["customer_average"]
    return amount / average
transaction = {
        "transaction_id": "TX001",
        "amount": 500,
        "customer_average": 450,
        "transactions_today": 2,
        "international": False
    }
ratio = amount_ratio(transaction)
print(ratio)

for a in transactions:
    ratio = amount_ratio(a)
    print(a["transaction_id"])
    print(ratio)
def analyze_transaction(transaction_id,country):
    print(transaction_id)
    print(country)
  
for a in transactions:
    analyze_transaction(a) 
