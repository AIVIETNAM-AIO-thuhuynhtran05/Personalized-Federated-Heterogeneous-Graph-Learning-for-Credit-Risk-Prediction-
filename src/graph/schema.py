"""Six node types; all graph construction happens inside one client."""
NODE_TABLES = {
    "customer": ("application_train", "SK_ID_CURR"),
    "bureau": ("bureau", "SK_ID_BUREAU"),
    "previous_application": ("previous_application", "SK_ID_PREV"),
    "installment": ("installments_payments", None),
    "pos_cash": ("POS_CASH_balance", None),
    "credit_card": ("credit_card_balance", None),
}

TRANSACTION_TYPES = ("installment", "pos_cash", "credit_card")
GRAPH_SCHEMA_VERSION = "orphan_fallback_v2"
