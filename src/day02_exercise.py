TransactionID = "TX5001"
Customer = "  vishnu vardhan  "
Amount = "4850.75"
Averagetransaction = "620.50"
Merchant = "APPLE ONLINE STORE"
Country = "US"
Transactionstoday = 11
International = True

print(Customer.strip())
print(Merchant.title())
Amount = float(Amount)
Averagetransaction = float(Averagetransaction)
amountratio = Amount / Averagetransaction
print(amountratio)

high_amount = amountratio > 5

high_velocity = Transactionstoday > 10
print(high_amount)
print(high_velocity)
print(International)
suspicious = high_amount and high_velocity and International
print(suspicious)
print(f"{amountratio:.2f}")

print("----- FraudLens Transaction Report -----")
print()
print(f"Transaction: {TransactionID}")
print(f"Customer: {Customer}")
print(f"Merchant: {Merchant}")
print(f"Amount: ${Amount:.2f}")
print(f"Customer Average: ${Averagetransaction:.2f}")
print(f"Amount Ratio: {amountratio:.2f}")
print(f"Transactions Today: {Transactionstoday}")
print()
print(f"High Amount: {high_amount}")
print(f"High Velocity: {high_velocity}")
print(f"International: {International}")
print(f"Suspicious: {suspicious}")