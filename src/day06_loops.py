transactions = [500, 1200, 6200, 75, 8500]
print(transactions[0])
print(transactions[1])
print(transactions[2])
print(transactions[3])
print(transactions[4])
for amount in transactions:
    print(amount)
for amount in transactions:
    print(f"Transaction: ${amount}")
    print("Processing transaction")
for amount in transactions:
    fee = amount * 0.2
    print(f"fee_amount:${fee:.2f}")
transactions = [500, 1200, 6200, 75, 8500]

for amount in transactions:

    if amount > 5000:
        print(f"${amount:.2f} → HIGH VALUE")

    else:
        print(f"${amount:.2f} → Normal")

transactions = [500, 1200, 6200, 75, 8500]

high_value_count = 0

for amount in transactions:

    if amount > 5000:
        high_value_count += 1

print(high_value_count)  

vishnu = [200,300,400]
vishu_amount = 0
for amount in vishnu:
    vishu_amount += amount
print(vishu_amount)

for number in range(5):
    print(number)
for number in range(1, 6):
    print(number)    

transaction = {
    "transaction_id": "TX1001",
    "amount": 6200,
    "merchant": "Apple",
    "international": True
}
for key in transaction:
    print(key)

for key, value in transaction.items():
    print(key, value)

transactions = [
    {
        "transaction_id": "TX001",
        "amount": 500,
        "merchant": "Amazon"
    },
    {
        "transaction_id": "TX002",
        "amount": 6200,
        "merchant": "Crypto Exchange"
    },
    {
        "transaction_id": "TX003",
        "amount": 1200,
        "merchant": "Walmart"
    }
]

for transaction in transactions:
    print(transaction["amount"])

for transaction in transactions:
    if transaction["amount"] > 5000:
        print("high value")
    else:
        print("Normal transaction")
#continue means: “Skip the remaining code for this value and move to the next value.”
amount = [200, -300, 7999, 888, 90849]
print(amount)
for amount in amount:
    if amount < 0:
        continue
    print(amount)
#break means Stop the entire loop
a = [200, 6000, 7000, 4000, 300,58585858]     
print(a)  
for a in a:
    if a > 10000:
        break
    print(a)

#while loops

b = 10
while b <= 10:
    print(b)
    b += 1



