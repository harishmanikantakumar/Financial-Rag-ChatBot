import pdfplumber
import re
import os


# ============================================================
# GENERAL METRIC MAP
# ============================================================

METRIC_MAP = {
    "revenue": [
        "total net sales",
        "net sales",
        "total revenues",
        "total revenue"
    ],

    "net_income": [
        "net income",
        "net earnings"
    ],

    "operating_income": [
        "operating income",
        "income from operations"
    ],

    "cost_of_revenue": [
        "total cost of sales",
        "total cost of revenue",
        "total cost of revenues"
    ],
}


# ============================================================
# COMPANY DETECTION
# ============================================================

def detect_company_from_path(pdf_path):

    filename = os.path.basename(pdf_path).lower()

    if "apple" in filename:
        return "apple"

    if "microsoft" in filename or filename.startswith("ms"):
        return "microsoft"

    if "tesla" in filename:
        return "tesla"

    return None


# ============================================================
# NUMBER EXTRACTION
# ============================================================

def extract_numbers(line):
    """
    Extract financial numbers such as:

        211,915
        198,270
        168,088
        72,361

    Also handles numbers without commas.
    """

    return re.findall(
        r'\b\d{1,3}(?:,\d{3})+\b|\b\d{4,6}\b',
        line
    )


# ============================================================
# MICROSOFT YEAR POSITION
# ============================================================

def get_microsoft_year_position(year):

    positions = {
        2023: 0,
        2022: 1,
        2021: 2
    }

    return positions.get(int(year))


# ============================================================
# MICROSOFT INCOME STATEMENT PAGE DETECTION
# ============================================================

def is_microsoft_income_statement_page(text):
    """
    Microsoft-specific page detection.

    We specifically want the page containing:

        INCOME STATEMENTS

        Year Ended June 30, 2023 2022 2021

    This prevents us from accidentally using Microsoft's
    Page 44 SUMMARY RESULTS table, which only contains
    2023 and 2022.
    """

    text_lower = text.lower()

    if "income statements" not in text_lower:
        return False

    if "year ended june 30, 2023 2022 2021" not in text_lower:
        return False

    return True


# ============================================================
# MICROSOFT-SPECIFIC EXTRACTION
# ============================================================

def extract_microsoft_metric(text, metric, target_year):
    """
    Microsoft-specific financial statement extraction.

    IMPORTANT:

    Microsoft does NOT use Apple's table structure.

    We therefore use Microsoft-specific labels and rules.

    Expected Microsoft table:

        Total revenue              211,915 198,270 168,088
        Total cost of revenue      65,863  62,650  52,232
        Operating income           88,523  83,383  69,916
        Net income                 72,361  72,738  61,271

    Column order:

        2023 -> index 0
        2022 -> index 1
        2021 -> index 2
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    metric = metric.lower()

    # --------------------------------------------------------
    # Microsoft exact financial-statement labels
    # --------------------------------------------------------

    metric_patterns = {

        "revenue": [
            r"^total revenue\b"
        ],

        "net_income": [
            r"^net income\b"
        ],

        "operating_income": [
            r"^operating income\b"
        ],

        "cost_of_revenue": [
            r"^total cost of revenue\b"
        ]
    }

    patterns = metric_patterns.get(
        metric,
        []
    )

    if not patterns:
        return None

    # --------------------------------------------------------
    # Microsoft annual column order
    #
    # 2023 | 2022 | 2021
    # --------------------------------------------------------

    year_idx = get_microsoft_year_position(
        target_year
    )

    if year_idx is None:
        return None

    # ========================================================
    # SEARCH ONLY EXACT MICROSOFT FINANCIAL ROW
    # ========================================================

    for line in lines:

        line_lower = line.lower().strip()

        matched = False

        for pattern in patterns:

            if re.search(
                pattern,
                line_lower
            ):
                matched = True
                break

        if not matched:
            continue

        # ====================================================
        # EXTRACT NUMBERS FROM SAME ROW
        # ====================================================

        numbers = extract_numbers(line)

        financial_numbers = []

        for number in numbers:

            value = float(
                number.replace(",", "")
            )

            # Ignore percentage artifacts
            # and small numbers.
            if value >= 100:

                financial_numbers.append(
                    value
                )

        # ====================================================
        # STRICT MICROSOFT RULE
        #
        # We require ALL THREE YEAR VALUES.
        #
        # This prevents:
        #
        # 2023 = 72,361
        # 2022 = 72,738
        # 2021 = WRONG
        #
        # ====================================================

        if len(financial_numbers) < 3:

            continue

        # ----------------------------------------------------
        # Select requested year
        # ----------------------------------------------------

        value = financial_numbers[year_idx]

        return {
            "value_millions": value,
            "value_billions": value / 1000.0,
            "matched_label": line
        }

    # --------------------------------------------------------
    # Nothing exact found.
    #
    # DO NOT GUESS.
    #
    # Main RAG pipeline can fall back to FAISS/BM25.
    # --------------------------------------------------------

    return None


# ============================================================
# APPLE EXTRACTOR
# ============================================================
#
# IMPORTANT:
#
# THIS IS THE APPLE LOGIC YOU WERE ALREADY USING.
#
# DO NOT CHANGE THIS LOGIC.
#
# ============================================================

def extract_apple_metric(text, metric, target_year):

    metric_keywords = METRIC_MAP.get(
        metric.lower(),
        [metric.lower()]
    )

    lines = text.split("\n")

    year_column_idx = None

    for line in lines:

        line_lower = line.lower()

        # ----------------------------------------------------
        # Existing Apple year detection
        # ----------------------------------------------------

        if (
            ("2023" in line and "2022" in line)
            or
            ("2023" in line and "2024" in line)
        ):

            years_found = [
                int(y)
                for y in re.findall(
                    r'\b(202[0-9])\b',
                    line
                )
            ]

            if target_year in years_found:

                year_column_idx = (
                    years_found.index(
                        target_year
                    )
                )

            continue

        # ----------------------------------------------------
        # Existing Apple metric matching
        # ----------------------------------------------------

        if any(
            kw in line_lower
            for kw in metric_keywords
        ):

            matches = extract_numbers(line)

            if matches:

                clean_nums = [
                    float(
                        m.replace(",", "")
                    )
                    for m in matches
                ]

                idx = (
                    year_column_idx
                    if (
                        year_column_idx is not None
                        and
                        year_column_idx < len(
                            clean_nums
                        )
                    )
                    else 0
                )

                value = clean_nums[idx]

                if value < 100:
                    continue

                return {
                    "value_millions": value,
                    "value_billions": value / 1000.0,
                    "matched_label": line.strip()
                }

    return None


# ============================================================
# MAIN PDF METRIC PARSER
# ============================================================

def parse_pdf_metric(
    pdf_path,
    metric,
    target_year
):

    # ========================================================
    # CHECK FILE
    # ========================================================

    if not os.path.exists(pdf_path):
        return None

    # ========================================================
    # DETECT COMPANY
    # ========================================================

    company = detect_company_from_path(
        pdf_path
    )

    if company is None:
        return None

    # ========================================================
    # MICROSOFT
    # ========================================================

    if company == "microsoft":

        try:

            with pdfplumber.open(
                pdf_path
            ) as pdf:

                # ------------------------------------------------
                # Microsoft PDF:
                #
                # Income Statements = Page 58
                #
                # We search a small page range instead of
                # scanning the complete 100+ page PDF.
                # ------------------------------------------------

                start_page = 55
                end_page = min(
                    61,
                    len(pdf.pages)
                )

                for page_index in range(
                    start_page,
                    end_page
                ):

                    page = pdf.pages[
                        page_index
                    ]

                    text = page.extract_text()

                    if not text:
                        continue

                    # ------------------------------------------------
                    # Only accept Microsoft's actual 3-year
                    # Income Statements page.
                    # ------------------------------------------------

                    if not is_microsoft_income_statement_page(
                        text
                    ):
                        continue

                    # ------------------------------------------------
                    # Microsoft-specific extraction
                    # ------------------------------------------------

                    result = extract_microsoft_metric(
                        text,
                        metric,
                        target_year
                    )

                    if result:

                        result["page"] = (
                            page_index + 1
                        )

                        result["company"] = (
                            "microsoft"
                        )

                        return result

        except Exception as e:

            print(
                f"Microsoft extraction error: {e}"
            )

        return None

    # ========================================================
    # APPLE
    # ========================================================

    if company == "apple":

        try:

            with pdfplumber.open(
                pdf_path
            ) as pdf:

                # ------------------------------------------------
                # KEEP APPLE BEHAVIOR AS-IS
                # ------------------------------------------------

                for page_number, page in enumerate(
                    pdf.pages,
                    start=1
                ):

                    text = page.extract_text()

                    if not text:
                        continue

                    text_lower = text.lower()

                    if (
                        "consolidated statements of operations"
                        not in text_lower
                        and
                        "consolidated statements of income"
                        not in text_lower
                        and
                        "statements of income"
                        not in text_lower
                    ):
                        continue

                    result = extract_apple_metric(
                        text,
                        metric,
                        target_year
                    )

                    if result:

                        result["page"] = (
                            page_number
                        )

                        result["company"] = (
                            "apple"
                        )

                        return result

        except Exception as e:

            print(
                f"Apple extraction error: {e}"
            )

        return None

    # ========================================================
    # TESLA
    # ========================================================

    if company == "tesla":

        # ----------------------------------------------------
        # Tesla rules will be implemented separately.
        #
        # DO NOT apply Apple or Microsoft rules to Tesla.
        # ----------------------------------------------------

        return None

    return None