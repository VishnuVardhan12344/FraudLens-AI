transaction_id = "TX7001"
customer = "Vishnu"
amount =400.00
customer_average = 500.00
transactions_today =2
international = False
new_device = False

amount_ratio = amount / customer_average


if amount_ratio > 5 and transactions_today > 10 and international:
    Risk_Level = "high risk"
    action = "Send for manual review"
elif  amount_ratio > 3 or transactions_today > 7 or new_device:
    Risk_Level = "medium risk"
    action = "Monitor transaction"
else:
    Risk_Level = "lowrisk"
    action = "Approve"

print(f"Risk Level: {Risk_Level}")

print("----- FraudLens Transaction Report -----")
print()
print(f"Transaction: {transaction_id}")
print(f"Customer: {customer}")
print(f"Amount: ${amount:.2f}")
print(f"Customer Average: ${customer_average:.2f}")
print(f"Amount Ratio: {amount_ratio:.2f}")
print(f"Transactions Today: {transactions_today}")
print(f" International : {international}")
print(f" new device : {new_device}")
print()
print(f"Risk Level: {Risk_Level}")
print(f"Action: {action}")
