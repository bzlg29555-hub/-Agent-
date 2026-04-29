import os
from typing import TypedDict, List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

# 配置 API Key (请替换为你的 Key)
os.environ["OPENAI_API_KEY"] = "your_api_key_here"

# --- 1. 定义状态机数据结构 ---
class TestState(TypedDict):
    task_description: str     # 任务描述
    test_plan: List[str]      # 测试计划
    execution_results: List[Dict] # 执行结果：[{case: "...", status: "pass/fail", log: "..."}]
    final_report: str         # 质量评估总结
    next_step: str            # 路由控制

# --- 2. 模拟工具函数 (模拟真实的游戏/产品操作) ---
def mock_app_executor(action: str):
    """
    模拟执行环境。在实际项目中，这里会调用 Selenium 或 Unity SDK。
    """
    if "special_char" in action.lower():
        return "Error: System crashed when entering special characters in username."
    return "Success: Action executed smoothly, UI updated."

# --- 3. 定义 Agent 节点 ---

llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

def planner_node(state: TestState):
    """【规划 Agent】生成测试用例"""
    prompt = f"针对以下功能描述，生成3条简短的自动化测试步骤：\n{state['task_description']}"
    res = llm.invoke([SystemMessage(content="你是一个资深QA，请只输出步骤列表，每行一个。"), HumanMessage(content=prompt)])
    steps = res.content.strip().split('\n')
    return {"test_plan": steps, "next_step": "executor"}

def executor_node(state: TestState):
    """【执行 Agent】解析用例并调用模拟工具"""
    results = []
    for step in state['test_plan']:
        # 模拟 Agent 理解步骤并执行
        log = mock_app_executor(step)
        status = "FAIL" if "Error" in log else "PASS"
        results.append({"case": step, "status": status, "log": log})
    return {"execution_results": results, "next_step": "auditor"}

def auditor_node(state: TestState):
    """【审计 Agent】综合分析结果，给出质量评分"""
    summary_input = "\n".join([f"用例: {r['case']} | 结果: {r['status']} | 日志: {r['log']}" for r in state['execution_results']])
    prompt = f"请根据以下测试日志，给产品出具一份简要的质量评估报告和改进建议：\n{summary_input}"
    res = llm.invoke([SystemMessage(content="你是一个质量审计专家。"), HumanMessage(content=prompt)])
    return {"final_report": res.content, "next_step": "end"}

# --- 4. 构建图拓扑结构 ---

workflow = StateGraph(TestState)

# 添加节点
workflow.add_node("planner", planner_node)
workflow.add_node("executor", executor_node)
workflow.add_node("auditor", auditor_node)

# 设置逻辑连线
workflow.set_entry_point("planner")
workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "auditor")
workflow.add_edge("auditor", END)

# 编译应用
app = workflow.compile()

# --- 5. 运行测试系统 ---
if __name__ == "__main__":
    initial_input = {
        "task_description": "注册模块：包含用户名输入（不支持特殊字符）、密码输入和提交按钮。",
        "execution_results": [],
        "test_plan": []
    }
    
    print("--- 自动化测试系统启动 ---")
    for event in app.stream(initial_input):
        for node, state in event.items():
            print(f"\n[节点: {node}] 已完成任务")
    
    # 打印最终生成的报告
    final_state = app.invoke(initial_input)
    print("\n" + "="*30)
    print("AI 质量评估报告")
    print("="*30)
    print(final_state["final_report"])
