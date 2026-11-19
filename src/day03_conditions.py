amount = 500
print(amount)
if amount > 3000:
    print("High-value transaction")
else:
        print("low-value transaction")

vishnu = 2508
if vishnu > 500:
    print("high")
elif vishnu > 300:
    print("medium")
else:
    print("low")    

amount_ratio = 7.8
transactions_today = 12
international = True
if amount_ratio > 5 and transactions_today > 10:
     print("suspicious activity")

if international == True:
    print("International transaction")


if amount_ratio > 5 and transactions_today > 10 and international:
    print("Transaction requires review")
else:
    print("No review required")


amount_ratio = 7.8
transactions_today = 12
international = True

if amount_ratio > 5 and transactions_today > 10 and international:
    risk_level = "HIGH"
elif amount_ratio > 3 or transactions_today > 7:
    risk_level = "MEDIUM"
else:
    risk_level = "LOW"

print(f"Risk Level: {risk_level}")

amount_ratio = 7.8
transactions_today = 12
international = True

if amount_ratio > 5:

    if international:
        print("High amount international transaction")

    else:
        print("High amount domestic transaction")

        