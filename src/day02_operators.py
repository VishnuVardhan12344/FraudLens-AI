transaction_amount = 2500
customer_average = 800

a = 10
b = 3

print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a // b) #a division operation that divides two numbers and rounds the result down to the nearest whole number (towards negative infinity)
print(a % b)
print(a ** b)

amount_ratio = transaction_amount / customer_average

print(amount_ratio)


fraud_score = 10

fraud_score += 20
 

print(fraud_score)

print(transaction_amount > 1000)

amount = 2500

print(amount > 1000)
print(amount < 1000)
print(amount == 2500)
print(amount != 2500)
print(amount >= 2500)
print(amount <= 100)


customer_name = "Vishnu"
merchant = "APPLE STORE"
country = "US"

message = customer_name + " made a transaction at " + merchant

print(message)
#For modern Python, you'll frequently use f-strings.
print(f"{customer_name} spent money at {merchant}")
amount = 2849.75

print(f"{customer_name} spent ${amount} at {merchant}")

print(merchant.upper())
print(merchant.lower())
print(merchant.title())


merchant1 = "   Apple Store   "
merchant1 = merchant1.strip()

print(merchant1)

description = "Online purchase at Apple Store"
print("Apple" in description)
print("Vishnu" in description)
print(description.upper())

merchant = "Amazon"

print(len(merchant))
print(merchant[0])
print(merchant[-1])
transaction_id = "TX9001"
print(transaction_id[0:2])#Start at index 0 and stop before index 2.
amount = "2849.75"
customer_average = "700.00"
amount = float(amount)
customer_average = float(customer_average)
amount_ratio = amount / customer_average

print(amount_ratio)