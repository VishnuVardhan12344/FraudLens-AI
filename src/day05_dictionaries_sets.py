
transaction = {
    "transaction_id": "TX1001",
    "customer": "Vishnu",
    "amount": 4850.75,
    "merchant": "Apple",
    "international": True
}
transaction["country"] = "US"
print(transaction)
transaction["amount"] = 5000.00
transaction["customer_average"] = 600
print(transaction)
amount_ratio = (
    transaction["amount"]
    / transaction["customer_average"]
)
transaction["amount_ratio"] = amount_ratio
print(transaction)
print(transaction.keys())
print(transaction.values())
print(transaction.items())
for key, value in transaction.items():
    print(key, value)
print("amount" in transaction)

transaction.pop("amount_ratio")
print(transaction)

transaction2 = {
    "transaction_id": "TX1001",
    "amount": 4850.75,

    "customer": {
        "customer_id": "C101",
        "name": "Vishnu",
        "country": "US"
    },

    "merchant": {
        "merchant_id": "M55",
        "name": "Apple",
        "category": "Electronics"
    }
}
print(transaction2["customer"]["name"])

transactions = [
    {
        "transaction_id": "TX001",
        "amount": 500,
        "merchant": "Amazon"
    },
    {
        "transaction_id": "TX002",
        "amount": 6200,
        "merchant": "Apple"
    },
    {
        "transaction_id": "TX003",
        "amount": 120,
        "merchant": "Walmart"
    }
]
print(transactions)
print(transactions[0])
print(transactions[0]["amount"])
print(transactions[1]["merchant"])

##sets
#set contain unqiue elements
merchants = {
    "Apple",
    "Amazon",
    "Apple",
    "Walmart",
    "Amazon"
}

merchants = {
    "Apple",
    "Amazon",
    "Apple",
    "Walmart",
    "Amazon"
}

print(merchants)
merchants.add("target")
print(merchants)
merchants.remove("Apple")
print(merchants)
##| Structure | Syntax | Main idea |

##| List | `[]` | Ordered collection |
##| Tuple | `()` | Fixed/immutable sequence |
##| Dictionary | `{key: value}` | Key-value data |
##| Set | `{value1, value2}` | Unique values |