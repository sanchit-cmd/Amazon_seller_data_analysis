from __future__ import annotations

from io import BytesIO
from typing import Any
import zipfile

import pandas as pd
import streamlit as st


class InputValidationError(ValueError):
    """Raised when an uploaded Amazon report is missing required data."""


def _read_tsv_file(file_bytes: bytes, label: str) -> pd.DataFrame:
    try:
        dataframe = pd.read_csv(
            BytesIO(file_bytes),
            sep="\t",
            encoding="ISO-8859-1",
            dtype="string",
        )
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise InputValidationError(f"Could not read the {label} as a tab-separated file.") from exc

    dataframe.columns = (
        dataframe.columns.astype(str).str.strip().str.replace(" ", "-", regex=False).str.casefold()
    )
    return dataframe


def _required_columns(dataframe: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise InputValidationError(
            f"The {label} is missing required column(s): {', '.join(missing)}."
        )


def _clean_text(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip()


def _sku_key(values: pd.Series) -> pd.Series:
    return _clean_text(values).str.casefold()


def calculate_results(
    orders_bytes: bytes,
    transactions_bytes: bytes,
    returns_bytes: bytes,
    sku_master_bytes: bytes,
    packaging_cost_per_order: float,
    ad_expenses: float,
    other_expenses: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Process the four Amazon input reports and return summary and detail tables."""
    orders = _read_tsv_file(orders_bytes, "orders report")
    _required_columns(
        orders,
        {"amazon-order-id", "sku", "quantity", "item-price"},
        "orders report",
    )
    orders = orders[["amazon-order-id", "sku", "quantity", "item-price"]].copy()
    orders["amazon-order-id"] = _clean_text(orders["amazon-order-id"])
    orders["sku"] = _clean_text(orders["sku"])
    orders["quantity"] = pd.to_numeric(orders["quantity"], errors="coerce")
    orders["item-price"] = pd.to_numeric(orders["item-price"], errors="coerce")
    orders = orders[
        orders["quantity"].ne(0)
        & orders["item-price"].notna()
        & orders["item-price"].ne(0)
    ].copy()

    if orders.empty:
        raise InputValidationError(
            "The orders report has no rows remaining after excluding zero quantities "
            "and missing or zero item prices."
        )
    if orders["quantity"].isna().any():
        raise InputValidationError(
            "Some order rows with a valid item price have a missing or non-numeric quantity."
        )
    if orders["amazon-order-id"].isna().any() or orders["amazon-order-id"].eq("").any():
        raise InputValidationError("Some included order rows are missing an Amazon Order ID.")
    if orders["sku"].isna().any() or orders["sku"].eq("").any():
        raise InputValidationError("Some included order rows are missing an SKU.")

    returns = _read_tsv_file(returns_bytes, "returns report")
    _required_columns(returns, {"order-id", "merchant-sku"}, "returns report")
    returns = returns[["order-id", "merchant-sku"]].dropna().copy()
    returns["order-id"] = _clean_text(returns["order-id"])
    returns["merchant-sku"] = _sku_key(returns["merchant-sku"])
    returns = returns[
        returns["order-id"].ne("") & returns["merchant-sku"].ne("")
    ]
    returned_pairs = set(zip(returns["order-id"], returns["merchant-sku"]))

    try:
        sku_master = pd.read_excel(BytesIO(sku_master_bytes), dtype={"SKU": "string"})
    except (ValueError, zipfile.BadZipFile, ImportError) as exc:
        raise InputValidationError("Could not read the SKU Master as an Excel workbook.") from exc
    sku_master.columns = sku_master.columns.astype(str).str.strip().str.casefold()
    _required_columns(sku_master, {"sku", "cost"}, "SKU Master")
    sku_master = sku_master[["sku", "cost"]].copy()
    sku_master["sku"] = _clean_text(sku_master["sku"])
    sku_master["Unit_Cost"] = pd.to_numeric(sku_master.pop("cost"), errors="coerce")
    sku_master = sku_master.dropna(subset=["sku"])
    sku_master = sku_master[sku_master["sku"].ne("")].copy()
    sku_master["sku_key"] = _sku_key(sku_master["sku"])

    duplicate_skus = sku_master.loc[
        sku_master["sku_key"].duplicated(keep=False), "sku"
    ].drop_duplicates()
    if not duplicate_skus.empty:
        examples = ", ".join(duplicate_skus.head(5).tolist())
        raise InputValidationError(
            "The SKU Master contains duplicate SKU entries, so costs are ambiguous. "
            f"Examples: {examples}."
        )

    if sku_master["Unit_Cost"].isna().any():
        invalid_cost_skus = sku_master.loc[
            sku_master["Unit_Cost"].isna(), "sku"
        ].head(5).tolist()
        examples = ", ".join(invalid_cost_skus)
        raise InputValidationError(
            "The SKU Master has blank or non-numeric costs. "
            f"Examples: {examples}."
        )

    orders["sku_key"] = _sku_key(orders["sku"])
    orders["is_returned"] = [
        int((order_id, sku) in returned_pairs)
        for order_id, sku in zip(orders["amazon-order-id"], orders["sku_key"])
    ]

    order_costs = orders.merge(
        sku_master[["sku_key", "Unit_Cost"]],
        on="sku_key",
        how="left",
        validate="many_to_one",
    )
    missing_cost_skus = sorted(
        order_costs.loc[order_costs["Unit_Cost"].isna(), "sku"].dropna().unique().tolist()
    )
    order_costs["Unit_Cost"] = order_costs["Unit_Cost"].fillna(0)

    try:
        transactions = pd.read_csv(
            BytesIO(transactions_bytes),
            skiprows=13,
            encoding="utf-8-sig",
            dtype="string",
        )
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise InputValidationError(
            "Could not read the Unified Transaction report. It should be a CSV with "
            "the report's 13 introductory rows followed by its column headers."
        ) from exc
    transactions.columns = transactions.columns.astype(str).str.strip().str.casefold()
    _required_columns(transactions, {"order id", "total"}, "Unified Transaction report")
    transactions = transactions[["order id", "total"]].copy()
    transactions["order id"] = _clean_text(transactions["order id"])
    transactions["total"] = pd.to_numeric(
        transactions["total"].astype("string").str.replace(",", "", regex=False),
        errors="coerce",
    )
    invalid_transaction_amounts = int(transactions["total"].isna().sum())
    transactions["total"] = transactions["total"].fillna(0)
    missing_transaction_ids = (
        transactions["order id"].isna() | transactions["order id"].eq("")
    )
    transactions = transactions.loc[~missing_transaction_ids].copy()
    order_payouts = (
        transactions.groupby("order id", as_index=False)["total"]
        .sum()
        .rename(columns={"order id": "amazon-order-id", "total": "Net_Payout"})
    )

    reconciliation = order_costs.merge(
        order_payouts,
        on="amazon-order-id",
        how="left",
        validate="many_to_one",
    )
    missing_payout_order_ids = sorted(
        reconciliation.loc[reconciliation["Net_Payout"].isna(), "amazon-order-id"]
        .dropna()
        .unique()
        .tolist()
    )
    reconciliation["Net_Payout"] = reconciliation["Net_Payout"].fillna(0)
    reconciliation["COGS"] = (
        (reconciliation["Unit_Cost"] * reconciliation["quantity"])
        .where(reconciliation["is_returned"].eq(0), 0.0)
        .astype(float)
    )

    order_summary = (
        reconciliation.groupby("amazon-order-id", as_index=False)
        .agg(
            Net_Payout=("Net_Payout", "first"),
            is_returned=("is_returned", "max"),
            Item_Lines=("sku", "size"),
            Returned_Item_Lines=("is_returned", "sum"),
            Total_Quantity=("quantity", "sum"),
            Total_COGS=("COGS", "sum"),
        )
    )

    total_net_payout = float(order_summary["Net_Payout"].sum())
    total_cogs = float(reconciliation["COGS"].sum())
    unique_order_count = int(order_summary["amazon-order-id"].nunique())
    total_packaging_cost = unique_order_count * packaging_cost_per_order
    net_profit = (
        total_net_payout
        - total_cogs
        - total_packaging_cost
        - ad_expenses
        - other_expenses
    )
    summary: dict[str, Any] = {
        "item_rows": len(reconciliation),
        "unique_orders": unique_order_count,
        "returned_items": int(reconciliation["is_returned"].sum()),
        "non_returned_items": int(reconciliation["is_returned"].eq(0).sum()),
        "total_net_payout": total_net_payout,
        "total_cogs": total_cogs,
        "packaging_cost_per_order": packaging_cost_per_order,
        "total_packaging_cost": total_packaging_cost,
        "ad_expenses": ad_expenses,
        "other_expenses": other_expenses,
        "net_profit": net_profit,
        "unmatched_sku_count": len(missing_cost_skus),
        "orders_without_payout_count": len(missing_payout_order_ids),
        "invalid_transaction_amounts": invalid_transaction_amounts,
        "transactions_without_order_id": int(missing_transaction_ids.sum()),
    }

    detail_columns = [
        "amazon-order-id",
        "sku",
        "quantity",
        "item-price",
        "is_returned",
        "Unit_Cost",
        "Net_Payout",
        "COGS",
    ]
    reconciliation = reconciliation[detail_columns].sort_values(
        ["amazon-order-id", "sku"], kind="stable"
    )
    order_summary = order_summary.sort_values("amazon-order-id", kind="stable")
    return (
        summary,
        reconciliation,
        order_summary,
        missing_cost_skus,
        missing_payout_order_ids,
    )


def _make_excel_download(
    reconciliation: pd.DataFrame, order_summary: pd.DataFrame, summary: dict[str, Any]
) -> bytes:
    output = BytesIO()
    summary_table = pd.DataFrame(
        [{"Metric": key, "Value": value} for key, value in summary.items()]
    )
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_table.to_excel(writer, sheet_name="Profit Summary", index=False)
        reconciliation.to_excel(writer, sheet_name="Item Details", index=False)
        order_summary.to_excel(writer, sheet_name="Order Summary", index=False)
    return output.getvalue()


def _format_rupees(value: float) -> str:
    return f"₹{value:,.2f}"


def main() -> None:
    st.set_page_config(page_title="Amazon Net Profit Calculator", page_icon="📊", layout="wide")
    st.title("Amazon Net Profit Calculator")
    st.caption(
        "Upload the same four raw Amazon reports used by the notebook. "
        "Returns are matched for each Order ID + SKU pair."
    )

    with st.expander("Input files and calculation method", expanded=True):
        st.markdown(
            """
            Upload: **Orders report** (`.txt`, tab-separated), **Unified Transaction report**
            (`.csv`, with 13 introductory rows), **Returns report** (`.tsv`), and
            **SKU Master** (`.xlsx`).

            Returned item lines are identified by matching both the order ID and SKU.
            Returned items are assigned zero COGS; other items use unit cost × quantity.
            Payout transactions are summed by order ID and counted once per order.
            """
        )

    with st.form("amazon_profit_inputs"):
        left, right = st.columns(2)
        with left:
            orders_file = st.file_uploader(
                "Orders report (.txt, tab-separated)", type=["txt", "tsv"], key="orders"
            )
            transactions_file = st.file_uploader(
                "Unified Transaction report (.csv)", type=["csv"], key="transactions"
            )
        with right:
            returns_file = st.file_uploader(
                "Returns report (.tsv)", type=["tsv", "txt"], key="returns"
            )
            sku_master_file = st.file_uploader(
                "SKU Master (.xlsx)", type=["xlsx"], key="sku_master"
            )

        st.subheader("Additional expenses (₹)")
        expense_columns = st.columns(3)
        with expense_columns[0]:
            packaging_cost = st.number_input(
                "Packaging cost per order", min_value=0.0, value=15.0, step=1.0
            )
        with expense_columns[1]:
            ad_expenses = st.number_input(
                "Monthly ad expenses", min_value=0.0, value=0.0, step=100.0
            )
        with expense_columns[2]:
            other_expenses = st.number_input(
                "Other expenses", min_value=0.0, value=0.0, step=100.0
            )

        calculate = st.form_submit_button("Calculate net profit", type="primary")

    if not calculate:
        return
    if not all((orders_file, transactions_file, returns_file, sku_master_file)):
        st.error("Please upload all four required reports before calculating.")
        return

    try:
        (
            summary,
            reconciliation,
            order_summary,
            missing_cost_skus,
            missing_payout_order_ids,
        ) = calculate_results(
            orders_file.getvalue(),
            transactions_file.getvalue(),
            returns_file.getvalue(),
            sku_master_file.getvalue(),
            packaging_cost,
            ad_expenses,
            other_expenses,
        )
    except (InputValidationError, pd.errors.ParserError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
        st.error(f"Could not calculate profit: {exc}")
        return

    st.success("Calculation completed.")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Net payout", _format_rupees(summary["total_net_payout"]))
    metric_columns[1].metric("Product cost (COGS)", _format_rupees(summary["total_cogs"]))
    metric_columns[2].metric(
        "Packaging cost", _format_rupees(summary["total_packaging_cost"])
    )
    metric_columns[3].metric(
        "Net profit / loss", _format_rupees(summary["net_profit"])
    )

    st.subheader("Calculation details")
    detail_columns = st.columns(4)
    detail_columns[0].metric("Unique orders", f"{summary['unique_orders']:,}")
    detail_columns[1].metric("Item rows", f"{summary['item_rows']:,}")
    detail_columns[2].metric("Returned items", f"{summary['returned_items']:,}")
    detail_columns[3].metric(
        "Non-returned items", f"{summary['non_returned_items']:,}"
    )
    st.caption(
        f"Profit = payout − COGS − packaging − ads − other expenses. "
        f"Ads: {_format_rupees(summary['ad_expenses'])}; "
        f"other expenses: {_format_rupees(summary['other_expenses'])}."
    )

    if missing_cost_skus:
        st.warning(
            f"{len(missing_cost_skus)} ordered SKU(s) were not found in the SKU Master. "
            "As in the notebook, their unit cost is treated as ₹0, so review this list "
            "before relying on the profit total."
        )
        st.dataframe(
            pd.DataFrame({"SKU missing from SKU Master": missing_cost_skus}),
            hide_index=True,
            use_container_width=True,
        )
    if missing_payout_order_ids:
        st.warning(
            f"{len(missing_payout_order_ids)} order(s) have no matching transaction "
            "payout and are counted as ₹0. Review these orders."
        )
        st.dataframe(
            pd.DataFrame({"Order ID without payout": missing_payout_order_ids}),
            hide_index=True,
            use_container_width=True,
        )
    if summary["invalid_transaction_amounts"]:
        st.warning(
            f"{summary['invalid_transaction_amounts']} transaction amount(s) were blank "
            "or non-numeric and are treated as ₹0, matching the notebook's conversion."
        )
    if summary["transactions_without_order_id"]:
        st.info(
            f"{summary['transactions_without_order_id']} transaction row(s) without an "
            "Order ID were excluded from order-level payouts."
        )

    item_tab, order_tab = st.tabs(["Item details", "Order summary"])
    with item_tab:
        st.dataframe(reconciliation, hide_index=True, use_container_width=True)
    with order_tab:
        st.dataframe(order_summary, hide_index=True, use_container_width=True)

    workbook = _make_excel_download(reconciliation, order_summary, summary)
    st.download_button(
        "Download results as Excel",
        data=workbook,
        file_name="amazon_profit_reconciliation.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    main()