# Audit dữ liệu theo ownership và số lần xuất hiện cặp khóa; đọc chunk và dùng SQLite để hạn chế RAM.

"""
Audit FK ownership and exact key-pair multiplicities using bounded-memory reads.

Mục tiêu:
- Kiểm tra việc chia dữ liệu Home Credit vào các client có đúng hay không.
- Kiểm tra Customer ownership, FK consistency và số lần xuất hiện của từng cặp khóa.
- Đảm bảo dữ liệu sau partition không bị mất, không bị dư, không bị gán sai client.
- Đọc dữ liệu theo chunk để giới hạn RAM.
"""

import argparse
import json
from pathlib import Path
import sqlite3
import tempfile
from datetime import datetime, timezone
import pandas as pd


# ============================================================
# 1. Khai báo các bảng relational và khóa liên kết tương ứng
# ============================================================

# Mỗi bảng sẽ được audit theo cặp:
# (SK_ID_CURR, relational_key)
#
# Ví dụ:
# bureau                 -> (SK_ID_CURR, SK_ID_BUREAU)
# previous_application   -> (SK_ID_CURR, SK_ID_PREV)
# installments_payments  -> (SK_ID_CURR, SK_ID_PREV)

RELATED = {
    "bureau": "SK_ID_BUREAU",
    "previous_application": "SK_ID_PREV",
    "installments_payments": "SK_ID_PREV",
    "POS_CASH_balance": "SK_ID_PREV",
    "credit_card_balance": "SK_ID_PREV"
}


def audit(source, clients, output, chunksize=100_000):

    
    # 2. Đọc manifest chứa thông tin các client
  

    manifest = json.loads(
        (clients / "partition_report.json").read_text()
    )

  
    # 3. Kiểm tra population Customer gốc
 
    # Chỉ đọc SK_ID_CURR để tiết kiệm bộ nhớ
    population = pd.read_csv(
        source / "application_train.csv",
        usecols=["SK_ID_CURR"]
    )

    # Điều kiện cần:
    # - SK_ID_CURR không được null
    # - SK_ID_CURR phải unique trong application_train
    if population.SK_ID_CURR.isna().any() or not population.SK_ID_CURR.is_unique:
        raise ValueError("Source Customer IDs invalid")

    # 4. Gom Customer từ tất cả client
   

    pieces = []

    for client in manifest["clients"]:

        frame = pd.read_csv(
            clients / client / "application_train.csv",
            usecols=["SK_ID_CURR"]
        )

        # Ghi lại customer này thực tế đang nằm trong client nào
        frame["client"] = client

        pieces.append(frame)

    actual = pd.concat(
        pieces,
        ignore_index=True
    )
   
    # 5. Audit Customer allocation
  
    customer_check = {

        # Tổng số customer trong source
        "source_rows": len(population),

        # Tổng số customer sau khi gom tất cả client
        "assigned_rows": len(actual),

        # Có customer nào bị null ID không?
        "null_ids": int(
            actual.SK_ID_CURR.isna().sum()
        ),

        # Có customer bị xuất hiện nhiều hơn 1 lần không?
        # Nếu có có thể nghĩa là cùng customer bị assign vào nhiều client
        "duplicate_ids": int(
            actual.SK_ID_CURR.duplicated().sum()
        ),

        # Customer có trong source nhưng bị mất sau partition
        "missing_ids": int(
            (~population.SK_ID_CURR.isin(actual.SK_ID_CURR)).sum()
        ),

        # Customer xuất hiện trong client nhưng không tồn tại trong source
        "extra_ids": int(
            (~actual.SK_ID_CURR.isin(population.SK_ID_CURR)).sum()
        )
    }

    # Nếu có bất kỳ lỗi nghiêm trọng nào ở Customer allocation
    # thì dừng pipeline ngay
    if any(
        customer_check[k]
        for k in (
            "null_ids",
            "duplicate_ids",
            "missing_ids",
            "extra_ids"
        )
    ):
        raise ValueError(
            f"Invalid Customer allocation: {customer_check}"
        )

    
    # 6. Tạo mapping SK_ID_CURR -> client thực tế
  
    # Ví dụ:
    #
    # SK_ID_CURR
    # 100002 -> client_000
    # 100003 -> client_002
    #
    # Mapping này được dùng để kiểm tra ownership của relational rows.

    owner = actual.set_index("SK_ID_CURR").client

    # 7. Audit assignments.csv
    

    assignments = pd.read_csv(
        clients / "assignments.csv"
    )

    # Convert client_id dạng số:
    #
    # 0 -> client_000
    # 1 -> client_001
    # 2 -> client_002

    assignment_names = assignments.client_id.map(
        lambda x: f"client_{int(x):03d}"
    )

    # Kiểm tra assignments.csv khai báo đúng client
    # so với dữ liệu thực tế hay không
    customer_check["assignment_mismatches"] = int(
        (
            assignments.SK_ID_CURR.map(owner)
            != assignment_names
        ).sum()
    )

    # Một Customer không được xuất hiện nhiều lần trong assignments.csv
    customer_check["assignment_duplicate_ids"] = int(
        assignments.SK_ID_CURR.duplicated().sum()
    )

    # Customer có trong source nhưng không có trong assignments.csv
    customer_check["assignment_missing_ids"] = int(
        (
            ~population.SK_ID_CURR.isin(
                assignments.SK_ID_CURR
            )
        ).sum()
    )

  
    # 8. Tạo mapping Previous Application -> Customer
    
    previous = pd.read_csv(
        source / "previous_application.csv",
        usecols=["SK_ID_CURR", "SK_ID_PREV"]
    )

    # SK_ID_PREV trong source phải:
    # - không null
    # - unique
    if (
        previous.SK_ID_PREV.isna().any()
        or not previous.SK_ID_PREV.is_unique
    ):
        raise ValueError(
            "Source Previous IDs invalid"
        )

    # Mapping:
    #
    # SK_ID_PREV -> SK_ID_CURR
    #
    # Ví dụ:
    # 200001 -> 100002
    #
    # Mapping này được dùng để kiểm tra installments/POS/credit_card
    # có trỏ về đúng Customer hay không.

    previous_owner = (
        previous
        .set_index("SK_ID_PREV")
        .SK_ID_CURR
    )

    
    # 9. Khởi tạo audit result
  
    result = {

        "checked_at_utc":
            datetime.now(timezone.utc).isoformat(),

        "source":
            str(source.resolve()),

        "clients_path":
            str(clients.resolve()),

        "num_clients":
            len(manifest["clients"]),

        "declared_strategy":
            manifest.get(
                "strategy",
                "not_recorded"
            ),

        "customer":
            customer_check,

        "tables": {},

        "scope":
            (
                "All rows; FK values, ownership and key-pair "
                "multiplicities. Feature values are not compared."
            )
    }

    # Tạo thư mục output nếu chưa tồn tại
    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

  
    # 10. Tạo SQLite tạm để lưu count
   
    #
    # Lý do dùng SQLite:
    # Các bảng như installments_payments có hàng triệu rows.
    #
    # Nếu giữ toàn bộ count trong RAM có thể rất tốn memory.
    #
    # SQLite giúp:
    # source_count - client_count
    # được lưu disk-based.

    with tempfile.TemporaryDirectory(
        prefix="relational_audit_",
        dir=output.parent
    ) as directory:

        connection = sqlite3.connect(
            str(
                Path(directory)
                / "counts.sqlite"
            )
        )

        try:

            # Tối ưu SQLite cho audit temporary database
            connection.execute(
                "PRAGMA journal_mode=OFF"
            )

            connection.execute(
                "PRAGMA synchronous=OFF"
            )

            connection.execute(
                "PRAGMA cache_size=-16000"
            )

           
            # 11. Audit từng relational table       

            for table, key in RELATED.items():

                print(
                    f"Auditing {table}...",
                    flush=True
                )

                # ---------------------------------------------
                # SQLite table dùng để lưu:
                #
                # (SK_ID_CURR, relational_key, delta)
                #
                # delta = source_count - client_count
                #
                # Nếu cuối cùng:
                # delta = 0 -> đúng
                # delta > 0 -> client bị thiếu record
                # delta < 0 -> client bị dư record
                # ---------------------------------------------

                connection.execute(
                    """
                    CREATE TABLE counts (
                        curr INTEGER,
                        rel INTEGER,
                        delta INTEGER,
                        PRIMARY KEY(curr, rel)
                    ) WITHOUT ROWID
                    """
                )

                # Summary tổng cho table hiện tại
                summary = {

                    "source_rows": 0,

                    # Rows trong source có Customer thuộc population
                    "eligible_source_rows": 0,

                    # Rows source có SK_ID_CURR không thuộc population
                    "excluded_source_rows": 0,

                    # Tổng rows đọc từ client
                    "client_rows": 0,

                    # Row nằm sai client
                    "wrong_client_rows": 0,

                    # Row có null ở SK_ID_CURR hoặc relational key
                    "null_key_rows": 0,

                    # SK_ID_PREV không tồn tại ngay trong source
                    "missing_previous_in_source": 0,

                    # SK_ID_PREV tồn tại trong source
                    # nhưng parent previous_application lại thiếu
                    # trong local client
                    "missing_previous_only_locally": 0,

                    # SK_ID_PREV tồn tại nhưng thuộc Customer khác
                    "previous_customer_mismatch": 0,

                    "clients": {}
                }

        
                # Helper function:
                # cập nhật multiplicity count trong SQLite              

                def accumulate(frame, sign):

                    # Fill null bằng -1 để vẫn có thể groupby
                    # và audit được null key nếu cần
                    grouped = (
                        frame
                        .fillna(-1)
                        .groupby(
                            ["SK_ID_CURR", key],
                            sort=False
                        )
                        .size()
                    )

                    # sign:
                    #
                    # +1 cho source
                    # -1 cho client
                    #
                    # Sau cùng:
                    # delta = source_count - client_count

                    connection.executemany(
                        """
                        INSERT INTO counts
                        VALUES (?, ?, ?)

                        ON CONFLICT(curr, rel)
                        DO UPDATE SET
                            delta =
                            delta + excluded.delta
                        """,
                        (
                            (
                                int(curr),
                                int(rel),
                                int(n) * sign
                            )
                            for (curr, rel), n
                            in grouped.items()
                        )
                    )

                    connection.commit()

              
                # 12. Đọc source theo chunks
               
                #
                # Không load toàn bộ table vào RAM.
                #
                # Ví dụ chunksize = 100,000:
                #
                # 100k rows -> process
                # 100k rows -> process
                # ...

                for chunk in pd.read_csv(
                    source / f"{table}.csv",
                    usecols=[
                        "SK_ID_CURR",
                        key
                    ],
                    chunksize=chunksize
                ):

                    # Chỉ xét rows có customer thuộc application_train
                    eligible = (
                        chunk.SK_ID_CURR.isin(
                            owner.index
                        )
                    )

                    summary["source_rows"] += len(
                        chunk
                    )

                    summary["eligible_source_rows"] += int(
                        eligible.sum()
                    )

                    summary["excluded_source_rows"] += int(
                        (~eligible).sum()
                    )

                    # Source được cộng +1 vào delta
                    accumulate(
                        chunk.loc[eligible],
                        1
                    )

              
                # 13. Audit từng client
           
                for client in manifest["clients"]:

                    detail = {

                        "rows": 0,

                        "wrong_client_rows": 0,

                        "null_key_rows": 0,

                        "missing_previous_in_source": 0,

                        "missing_previous_only_locally": 0,

                        "previous_customer_mismatch": 0
                    }

                    # Danh sách SK_ID_PREV thực sự tồn tại
                    # trong previous_application của local client
                    local_previous = pd.Index(
                        pd.read_csv(
                            clients
                            / client
                            / "previous_application.csv",
                            usecols=["SK_ID_PREV"]
                        ).SK_ID_PREV
                    )

                    # Đọc relational table của client theo chunk
                    for chunk in pd.read_csv(
                        clients
                        / client
                        / f"{table}.csv",
                        usecols=[
                            "SK_ID_CURR",
                            key
                        ],
                        chunksize=chunksize
                    ):

                        detail["rows"] += len(
                            chunk
                        )

                        # -------------------------------------
                        # A. Kiểm tra row có nằm đúng client?
                        # -------------------------------------
                        #
                        # Ví dụ:
                        #
                        # Customer 100002
                        # owner = client_000
                        #
                        # nhưng row lại nằm ở client_003
                        # -> wrong_client_rows += 1

                        detail["wrong_client_rows"] += int(
                            (
                                chunk.SK_ID_CURR.map(
                                    owner
                                )
                                != client
                            ).sum()
                        )

                        # -------------------------------------
                        # B. Kiểm tra null key
                        # -------------------------------------
                        #
                        # Nếu SK_ID_CURR hoặc key bị null
                        # thì relational link có thể bị broken.

                        detail["null_key_rows"] += int(
                            chunk
                            .isna()
                            .any(axis=1)
                            .sum()
                        )

                        # -------------------------------------
                        # C. Audit SK_ID_PREV
                        # -------------------------------------
                        #
                        # Chỉ áp dụng cho:
                        # installments_payments
                        # POS_CASH_balance
                        # credit_card_balance
                        #
                        # bureau không dùng SK_ID_PREV.
                        # previous_application chính là parent table.

                        if table not in (
                            "bureau",
                            "previous_application"
                        ):

                            # Tìm Customer owner của SK_ID_PREV
                            mapped = (
                                chunk.SK_ID_PREV.map(
                                    previous_owner
                                )
                            )

                            # SK_ID_PREV có tồn tại
                            # trong source previous_application không?
                            exists = (
                                chunk.SK_ID_PREV.isin(
                                    previous_owner.index
                                )
                            )

                            # ---------------------------------
                            # C1. Source orphan
                            # ---------------------------------
                            #
                            # Child record trỏ đến SK_ID_PREV
                            # không tồn tại ngay trong source.

                            detail[
                                "missing_previous_in_source"
                            ] += int(
                                (~exists).sum()
                            )

                            # ---------------------------------
                            # C2. Local missing parent
                            # ---------------------------------
                            #
                            # SK_ID_PREV tồn tại trong source,
                            # nhưng previous_application tương ứng
                            # không tồn tại trong client hiện tại.
                            #
                            # Đây là lỗi partition nghiêm trọng.

                            detail[
                                "missing_previous_only_locally"
                            ] += int(
                                (
                                    exists
                                    &
                                    ~chunk.SK_ID_PREV.isin(
                                        local_previous
                                    )
                                ).sum()
                            )

                            # ---------------------------------
                            # C3. Previous Application
                            #     thuộc sai Customer
                            # ---------------------------------
                            #
                            # Ví dụ:
                            #
                            # installment row:
                            # SK_ID_CURR = 100002
                            # SK_ID_PREV = 200005
                            #
                            # nhưng source mapping:
                            # 200005 -> 100010
                            #
                            # => relational inconsistency

                            detail[
                                "previous_customer_mismatch"
                            ] += int(
                                (
                                    exists
                                    &
                                    (
                                        mapped
                                        != chunk.SK_ID_CURR
                                    )
                                ).sum()
                            )

                        # Client được trừ -1 khỏi count
                        accumulate(
                            chunk,
                            -1
                        )

                
                    # 14. Cộng detail từng client
                    #     vào summary của table
                    
                    summary["client_rows"] += detail[
                        "rows"
                    ]

                    for field in detail:

                        if field != "rows":

                            summary[field] += detail[
                                field
                            ]

                    summary["clients"][
                        client
                    ] = detail

         
                # 15. So sánh source và clients
                #     bằng exact multiplicity
                              #
                # delta > 0:
                # source có nhiều occurrences hơn clients
                # -> missing
                #
                # delta < 0:
                # clients có nhiều hơn source
                # -> extra

                missing, extra = (
                    connection.execute(
                        """
                        SELECT
                            COALESCE(
                                SUM(
                                    CASE
                                        WHEN delta > 0
                                        THEN delta
                                        ELSE 0
                                    END
                                ),
                                0
                            ),
                            COALESCE(
                                SUM(
                                    CASE
                                        WHEN delta < 0
                                        THEN -delta
                                        ELSE 0
                                    END
                                ),
                                0
                            )
                        FROM counts
                        """
                    ).fetchone()
                )

                summary[
                    "missing_key_occurrences"
                ] = missing

                summary[
                    "extra_key_occurrences"
                ] = extra

                # Tỷ lệ child rows trỏ tới previous
                # không tồn tại ngay trong source
                summary[
                    "missing_previous_source_rate"
                ] = (
                    summary[
                        "missing_previous_in_source"
                    ]
                    /
                    max(
                        1,
                        summary["client_rows"]
                    )
                )

           
                # 16. Quyết định table có pass partition audit?
                          #
                # Table PASS nếu:
                #
                # 1. Không có row nằm sai client
                # 2. Không có null relational key
                # 3. Không có local missing parent
                # 4. Không mất occurrences
                # 5. Không dư occurrences

                summary[
                    "partition_pass"
                ] = not any(
                    summary[field]
                    for field in (
                        "wrong_client_rows",
                        "null_key_rows",
                        "missing_previous_only_locally",
                        "missing_key_occurrences",
                        "extra_key_occurrences"
                    )
                )

                # Lưu kết quả table
                result["tables"][
                    table
                ] = summary

                # Xóa temporary count table
                # trước khi audit bảng tiếp theo
                connection.execute(
                    "DROP TABLE counts"
                )

                connection.commit()

                print(
                    f"  {summary['client_rows']} rows; "
                    f"partition_pass={summary['partition_pass']}; "
                    f"source orphans="
                    f"{summary['missing_previous_in_source']}",
                    flush=True
                )

                # Ghi intermediate result
                # để nếu process bị dừng giữa chừng
                # vẫn còn audit của các table đã chạy
                output.write_text(
                    json.dumps(
                        result,
                        indent=2
                    ),
                    encoding="utf-8"
                )

        finally:

            connection.close()

   
    # 17. Global partition result

    # Toàn bộ partition PASS khi:
    #
    # - tất cả relational tables PASS
    # - assignments.csv không mismatch
    # - không duplicate assignment
    # - không missing assignment

    result[
        "partition_pass"
    ] = (

        all(
            table_result[
                "partition_pass"
            ]
            for table_result
            in result["tables"].values()
        )

        and not any(
            customer_check[k]
            for k in (
                "assignment_mismatches",
                "assignment_duplicate_ids",
                "assignment_missing_ids"
            )
        )
    )

    # 18. Kiểm tra Customer ↔ Previous consistency
      #
    # True nếu không có SK_ID_PREV nào
    # bị liên kết với sai SK_ID_CURR.

    result[
        "customer_consistent_previous_links"
    ] = all(

        table_result[
            "previous_customer_mismatch"
        ] == 0

        for table_result
        in result["tables"].values()
    )

   
    # 19. Lưu report cuối cùng
  

    output.write_text(
        json.dumps(
            result,
            indent=2
        ),
        encoding="utf-8"
    )

    return result



# 20. Command-line interface


if __name__ == "__main__":

    # Project root
    root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # Source relational tables
    parser.add_argument(
        "--source",
        type=Path,
        default=root
        / "data/interim/tables"
    )

    # Folder chứa các client sau partition
    parser.add_argument(
        "--clients",
        type=Path,
        default=root
        / "data/processed/clients"
    )

    # Audit output report
    parser.add_argument(
        "--output",
        type=Path,
        default=root
        / "results/relational_audit.json"
    )

    args = parser.parse_args()

    # Chạy audit
    audit(**vars(args))

