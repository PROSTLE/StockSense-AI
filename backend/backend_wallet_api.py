# Add these endpoints to your FastAPI backend (main.py or similar)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import time

app = FastAPI()

# In-memory wallet balance (replace with database in production)
wallet_data = {
    "balance": 10000.00,
    "transactions": []
}

class AddMoneyRequest(BaseModel):
    payment_id: str
    amount: float

class WithdrawRequest(BaseModel):
    amount: float

@app.get("/api/wallet/balance")
async def get_wallet_balance():
    """Get current wallet balance"""
    return {
        "balance": wallet_data["balance"],
        "status": "success"
    }

@app.post("/api/wallet/add")
async def add_money(request: AddMoneyRequest):
    """
    Add money to wallet after successful Razorpay payment
    
    In production, you should:
    1. Verify the payment with Razorpay API using payment_id
    2. Check payment status and amount
    3. Store transaction in database
    """
    
    # TODO: Verify payment with Razorpay
    # For now, we'll trust the frontend (NOT recommended for production)
    
    if request.amount <= 0:
        raise HTTPException(status_code=400, detail="Invalid amount")
    
    # Add money to wallet
    wallet_data["balance"] += request.amount
    
    # Store transaction
    transaction = {
        "type": "credit",
        "amount": request.amount,
        "payment_id": request.payment_id,
        "timestamp": time.time(),
        "status": "success"
    }
    wallet_data["transactions"].append(transaction)
    
    return {
        "status": "success",
        "message": f"₹{request.amount} added successfully",
        "new_balance": wallet_data["balance"],
        "transaction": transaction
    }

@app.post("/api/wallet/withdraw")
async def withdraw_money(request: WithdrawRequest):
    """
    Withdraw money from wallet
    
    In production, you should:
    1. Verify user authentication
    2. Check withdrawal limits
    3. Process actual bank transfer
    4. Store transaction in database
    """
    
    if request.amount <= 0:
        raise HTTPException(status_code=400, detail="Invalid amount")
    
    # Check sufficient balance
    if wallet_data["balance"] < request.amount:
        raise HTTPException(status_code=400, detail="Insufficient balance")
    
    # Deduct money from wallet
    wallet_data["balance"] -= request.amount
    
    # Store transaction
    transaction = {
        "type": "debit",
        "amount": request.amount,
        "timestamp": time.time(),
        "status": "success"
    }
    wallet_data["transactions"].append(transaction)
    
    return {
        "status": "success",
        "message": f"₹{request.amount} withdrawn successfully",
        "new_balance": wallet_data["balance"],
        "transaction": transaction
    }

@app.get("/api/wallet/transactions")
async def get_transactions():
    """Get all wallet transactions"""
    return {
        "transactions": wallet_data["transactions"],
        "status": "success"
    }

# Example of how to verify Razorpay payment (add this to your add_money endpoint)
"""
import razorpay

razorpay_client = razorpay.Client(auth=("YOUR_KEY_ID", "YOUR_KEY_SECRET"))

def verify_razorpay_payment(payment_id: str, amount: float):
    try:
        # Fetch payment details
        payment = razorpay_client.payment.fetch(payment_id)
        
        # Verify payment status
        if payment['status'] != 'captured':
            return False
        
        # Verify amount (Razorpay stores in paise)
        if payment['amount'] != int(amount * 100):
            return False
        
        return True
    except Exception as e:
        print(f"Payment verification failed: {e}")
        return False
"""
