"""Schema của heterogeneous graph Home Credit.

    customer ──has_bureau──▶ bureau
    customer ──has_prev────▶ prev ──has_installment──▶ installment
                                  ──has_pos──────────▶ pos
                                  ──has_cc───────────▶ cc
Mỗi cạnh có thêm cạnh ngược (rev_*) để message passing hai chiều.
bureau_balance không là node riêng: được aggregate thành đặc trưng của node bureau.
"""

CUSTOMER = "customer"
BUREAU = "bureau"
PREV = "prev"
INSTALLMENT = "installment"
POS = "pos"
CC = "cc"

NODE_TYPES = [CUSTOMER, BUREAU, PREV, INSTALLMENT, POS, CC]

# node con -> (node cha, tên quan hệ, bảng gốc, khóa ngoại tới cha)
PARENT = {
    BUREAU: (CUSTOMER, "has_bureau", "bureau", "SK_ID_CURR"),
    PREV: (CUSTOMER, "has_prev", "previous_application", "SK_ID_CURR"),
    INSTALLMENT: (PREV, "has_installment", "installments_payments", "SK_ID_PREV"),
    POS: (PREV, "has_pos", "POS_CASH_balance", "SK_ID_PREV"),
    CC: (PREV, "has_cc", "credit_card_balance", "SK_ID_PREV"),
}

EVENT_TYPES = [INSTALLMENT, POS, CC]


def edge_types() -> list[tuple[str, str, str]]:
    out = []
    for child, (parent, rel, _, _) in PARENT.items():
        out.append((parent, rel, child))
        out.append((child, f"rev_{rel}", parent))
    return out


# Cột phân loại của từng node type (còn lại là số)
CATEGORICAL = {
    BUREAU: ["CREDIT_ACTIVE", "CREDIT_CURRENCY", "CREDIT_TYPE"],
    PREV: ["NAME_CONTRACT_TYPE", "NAME_CONTRACT_STATUS", "NAME_PAYMENT_TYPE",
           "CODE_REJECT_REASON", "NAME_CLIENT_TYPE", "NAME_PORTFOLIO", "NAME_PRODUCT_TYPE",
           "CHANNEL_TYPE", "NAME_YIELD_GROUP", "PRODUCT_COMBINATION", "NAME_GOODS_CATEGORY",
           "NAME_CASH_LOAN_PURPOSE", "NAME_SELLER_INDUSTRY", "NAME_TYPE_SUITE",
           "WEEKDAY_APPR_PROCESS_START", "FLAG_LAST_APPL_PER_CONTRACT"],
    POS: ["NAME_CONTRACT_STATUS"],
    CC: ["NAME_CONTRACT_STATUS"],
    INSTALLMENT: [],
}
