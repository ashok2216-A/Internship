import pandas as pd


# 1. Sample dataset
data = {
    "messages": [
        {"message": "When will my order be delivered?", "category": "Inquiry"},
        {"message": "I want to know the status of my refund.", "category": "Inquiry"},
        {"message": "The product arrived damaged.", "category": "Complaint"},
        {"message": "The delivery is delayed again!", "category": "Complaint"},
        {"message": "Great customer service!", "category": "Feedback"},
        {"message": "I love your quick response time!", "category": "Feedback"},
        {"message": "Can you help me change my address?", "category": "Request"},
        {"message": "Please update my phone number.", "category": "Request"},
        {"message": "Where can I track my shipment?", "category": "Inquiry"},
        {"message": "The item I received is incorrect.", "category": "Complaint"},
        {"message": "Kudos to your team!", "category": "Feedback"},
        {"message": "I would like to cancel my order.", "category": "Request"},
        {"message": "Why is my payment still pending?", "category": "Inquiry"},
        {"message": "Received wrong item, very disappointed.", "category": "Complaint"},
        {"message": "Amazing shopping experience!", "category": "Feedback"},
        {"message": "Can you provide an estimated delivery date for my order?", "category": "Inquiry"},
        {"message": "Has my payment been successfully processed?", "category": "Inquiry"},
        {"message": "Is this item available for purchase?", "category": "Inquiry"},
        {"message": "What are the warranty details for this product?", "category": "Inquiry"},
        {"message": "How do I apply a discount code to my purchase?", "category": "Inquiry"},
        {"message": "My package arrived late, and I am not happy about it.", "category": "Complaint"},
        {"message": "The quality of the item does not match the description.", "category": "Complaint"},
        {"message": "I received a defective product—what should I do next?", "category": "Complaint"},
        {"message": "I was charged incorrectly—please fix this issue.", "category": "Complaint"},
        {"message": "The customer service response time is too slow.", "category": "Complaint"},
        {"message": "Your platform is user-friendly and easy to navigate!", "category": "Feedback"},
        {"message": "I appreciate how seamless the checkout process was.", "category": "Feedback"},
        {"message": "The packaging was excellent—no damages at all!", "category": "Feedback"},
        {"message": "Your team handled my issue efficiently—thank you!", "category": "Feedback"},
        {"message": "I'm impressed by your wide range of product choices.", "category": "Feedback"},
        {"message": "I need assistance with updating my billing information.", "category": "Request"},
        {"message": "Can I reschedule my delivery for a different date?", "category": "Request"},
        {"message": "Please help me apply a gift card to my order.", "category": "Request"},
        {"message": "I want to change the color of the item I ordered.", "category": "Request"},
        {"message": "Can I add an extra item to my existing order?", "category": "Request"}
    ]
}

# 2. Convert to DataFrame
df = pd.DataFrame(data["messages"])

# 3. Save DataFrame to CSV
df.to_csv("data.csv", index=False)