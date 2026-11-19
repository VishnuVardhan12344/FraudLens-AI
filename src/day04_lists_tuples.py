transactions = [500, 1200, 750, 5000]
print(transactions)
print(type(transactions))
transaction1 = [
    "TX1001",
    4850.75,
    "Apple",
    True
]

print(transaction1)
print(transactions[0])
print(len(transactions))
print(transactions[3:])
print(transactions[-3:])
transactions.append(9000)
print(transactions)
transactions.insert(4,8000)
print(transactions)
transactions.remove(8000)
print(transactions)
transactions.pop(2)
print(transactions)
print(600 in transactions)

if 600 in transactions:
    print("high risk")
else:
    print("low_risk")

print(len(transactions))
print(sum(transactions))
print(min(transactions))
print(max(transactions))

average = sum(transactions) / len(transactions)

print(average)
transactions.sort()
print(transactions)
transactions.sort(reverse = True)
print(transactions)

transactions2 = [
    ["TX001", 500, "Amazon"],
    ["TX002", 2500, "Apple"],
    ["TX003", 100, "Walmart"]
]
print(transactions2)
print(transactions2[0][1])
