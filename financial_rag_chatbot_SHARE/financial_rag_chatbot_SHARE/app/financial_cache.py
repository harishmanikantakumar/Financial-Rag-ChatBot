# app/financial_cache.py

FINANCIAL_DATA_CACHE = {
    "apple": {
        2023: {
            "revenue": {"millions": 383285, "label": "Total net sales 383,285"},
            "cost_of_revenue": {"millions": 214137, "label": "Total cost of sales 214,137"},
            "net_income": {"millions": 96995, "label": "Net income 96,995"},
            "operating_income": {"millions": 114301, "label": "Operating income 114,301"},
            "gross_margin": {"millions": 169148, "label": "Gross margin 169,148"}
        },
        2022: {
            "revenue": {"millions": 394328, "label": "Total net sales 394,328"},
            "cost_of_revenue": {"millions": 223546, "label": "Total cost of sales 223,546"},
            "net_income": {"millions": 99803, "label": "Net income 99,803"},
            "operating_income": {"millions": 119437, "label": "Operating income 119,437"}
        },
        2021: {
            "revenue": {"millions": 365817, "label": "Total net sales 365,817"},
            "cost_of_revenue": {"millions": 212981, "label": "Total cost of sales 212,981"},
            "net_income": {"millions": 94680, "label": "Net income 94,680"},
            "operating_income": {"millions": 108949, "label": "Operating income 108,949"}
        }
    },
    "microsoft": {
        2023: {
            "revenue": {"millions": 211915, "label": "Total revenue 211,915"},
            "cost_of_revenue": {"millions": 65863, "label": "Total cost of revenue 65,863"},
            "net_income": {"millions": 72361, "label": "Net income 72,361"},
            "operating_income": {"millions": 88523, "label": "Operating income 88,523"}
        },
        2022: {
            "revenue": {"millions": 198270, "label": "Total revenue 198,270"},
            "cost_of_revenue": {"millions": 62650, "label": "Total cost of revenue 62,650"},
            "net_income": {"millions": 72738, "label": "Net income 72,738"},
            "operating_income": {"millions": 83383, "label": "Operating income 83,383"}
        },
        2021: {
            "revenue": {"millions": 168088, "label": "Total revenue 168,088"},
            "cost_of_revenue": {"millions": 52230, "label": "Total cost of revenue 52,230"},
            "net_income": {"millions": 61271, "label": "Net income 61,271"},
            "operating_income": {"millions": 69916, "label": "Operating income 69,916"}
        }
    },
    "tesla": {
        2023: {
            "revenue": {"millions": 96773, "label": "Total revenues 96,773"},
            "cost_of_revenue": {"millions": 79113, "label": "Total cost of revenues 79,113"},
            "net_income": {"millions": 14974, "label": "Net income 14,974"},
            "operating_income": {"millions": 8891, "label": "Income from operations 8,891"},
            "gross_profit": {"millions": 17660, "label": "Total gross profit 17,660"}
        },
        2022: {
            "revenue": {"millions": 81462, "label": "Total revenues 81,462"},
            "cost_of_revenue": {"millions": 60609, "label": "Total cost of revenues 60,609"},
            "net_income": {"millions": 12587, "label": "Net income 12,587"},
            "operating_income": {"millions": 13656, "label": "Income from operations 13,656"},
            "gross_profit": {"millions": 20853, "label": "Total gross profit 20,853"}
        },
        2021: {
            "revenue": {"millions": 53823, "label": "Total revenues 53,823"},
            "cost_of_revenue": {"millions": 40217, "label": "Total cost of revenues 40,217"},
            "net_income": {"millions": 5519, "label": "Net income 5,519"},
            "operating_income": {"millions": 6523, "label": "Income from operations 6,523"},
            "gross_profit": {"millions": 13606, "label": "Total gross profit 13,606"}
        }
    }
}


def get_cached_metric(company: str, metric: str, year: int):
    """Fast O(1) financial metric lookup function."""
    if not company or not metric or not year:
        return None

    comp_data = FINANCIAL_DATA_CACHE.get(company.lower(), {})
    year_data = comp_data.get(int(year), {})
    metric_data = year_data.get(metric.lower(), None)

    if metric_data:
        val_m = metric_data["millions"]
        return {
            "value_millions": val_m,
            "value_billions": val_m / 1000.0,
            "matched_label": metric_data["label"],
        }
    return None