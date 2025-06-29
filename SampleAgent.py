from langgraph.graph import StateGraph, END  # type: ignore
from typing import TypedDict, List
import json
from langsmith import traceable

# library i used

class BusinessData(TypedDict):
    '''For better understanding of the business data structure'''
    daily_revenue: float
    daily_cost: float
    number_of_customers: int
    previous_day_revenue: float
    previous_day_cost: float

class BusinessOutput(TypedDict):
    '''For better understanding of the business output structure'''
    profit: float
    cac: float
    revenue_change_pct: float
    cost_change_pct: float
    alerts: List[str]
    recommendations: List[str]

@traceable(name="Input Node")
def input_node(state: dict) -> dict:
    return state

@traceable(name="Processing Node")
def processing_node(state: dict) -> dict:
    data: BusinessData = state["input"]
    profit = data["daily_revenue"] - data["daily_cost"]
    cac = data["daily_cost"] / data["number_of_customers"]
    revenue_change_pct = ((data["daily_revenue"] - data["previous_day_revenue"]) / data["previous_day_revenue"]) * 100
    cost_change_pct = ((data["daily_cost"] - data["previous_day_cost"]) / data["previous_day_cost"]) * 100
    state["metrics"] = {
        "profit": profit,
        "cac": cac,
        "revenue_change_pct": revenue_change_pct,
        "cost_change_pct": cost_change_pct,
    }
    return state

@traceable(name="Recommendation Node")
def recommendation_node(state: dict) -> dict:
    metrics = state["metrics"]
    alerts = []
    recommendations = []
    if metrics["profit"] < 0:
        alerts.append("Negative profit")
        recommendations.append("Reduce costs to increase profitability")
    if metrics["cac"] > 0 and metrics["cost_change_pct"] > 20:
        alerts.append("CAC increased significantly")
        recommendations.append("Review marketing campaigns due to CAC spike")
    if metrics["revenue_change_pct"] > 10:
        recommendations.append("Consider increasing advertising budget")
    state["output"] = {
        "profit": metrics["profit"],
        "cac": metrics["cac"],
        "revenue_change_pct": metrics["revenue_change_pct"],
        "cost_change_pct": metrics["cost_change_pct"],
        "alerts": alerts,
        "recommendations": recommendations,
    }
    return state

def build_graph():
    """Build a simple graph"""
    builder = StateGraph(dict) 
    builder.add_node("input_node", input_node)
    builder.add_node("processing_node", processing_node)
    builder.add_node("recommendation_node", recommendation_node)
    builder.set_entry_point("input_node")
    builder.add_edge("input_node", "processing_node")
    builder.add_edge("processing_node", "recommendation_node")
    builder.add_edge("recommendation_node", END)
    return builder.compile()

@traceable(name="Business Agent Run")
def run_graph(input_data):
    graph = build_graph()
    return graph.invoke(input_data)

def test_agent():
    test_input = {
        "input": {
            "daily_revenue": 1500,
            "daily_cost": 1700,
            "number_of_customers": 50,
            "previous_day_revenue": 1000,
            "previous_day_cost": 1000
        }
    }

    result = run_graph(test_input)
    output = result["output"]

    assert output["profit"] == -200, "Profit calculation failed"
    assert "Negative profit" in output["alerts"], "Missing alert"
    print("Test passed")
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    sample = {
        "input": {
            "daily_revenue": 1200,
            "daily_cost": 1000,
            "number_of_customers": 50,
            "previous_day_revenue": 1000,
            "previous_day_cost": 800
        }
    }
    output = run_graph(sample)
    print("Sample run output:")
    print(json.dumps(output["output"], indent=2))
    
    print("\n test...")
    test_agent()
